# Required Libraries
import pytesseract
import os
from PIL import Image
from pdf2image import convert_from_path
import json
import re
import requests
import numpy as np
import logging
from sklearn.ensemble import IsolationForest
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.document_loaders.image import UnstructuredImageLoader
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
import streamlit as st

# Setup Logging
logging.basicConfig(filename='audit_trail.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Set API Key from Streamlit Secrets
llm = ChatOpenAI(model="gpt-4o", openai_api_key=st.secrets["OPENAI_API_KEY"])

#----------- Safe Json Output -------
def safe_json(obj):
    try:
        return json.dumps(obj.dict())
    except AttributeError:
        return json.dumps(obj)

# ----------- Global Pydantic Schema -----------
class BankDetails(BaseModel):
    account_holder_name: str | None = Field(..., description="The full legal name of the individual or entity that owns the bank account, as mentioned in the document.")
    account_holder_account_number: str | None = Field(..., description="The unique number assigned by the bank to identify the account holder’s bank account.")
    account_holder_address: str | None = Field(..., description="The complete residential or business address of the account holder, including street, unit, city, state, and postal code as available.")
    routing_number: str | None = Field(..., description="The bank’s routing number used to identify the financial institution in a transaction. This may also be referred to as a bank code or ABA number.")
    bank_name: str | None = Field(..., description="The official name of the bank or financial institution as stated in the document. This could appear near the address or as a logo/header.")
    bank_address: str | None = Field(..., description="The full address of the bank branch, including street, city, and state, as mentioned in the document.")


# ----------- Agent 1: Document Extractor (OCR + Layout) -----------

def extract_text_from_document(uploaded_file): 
    try:
        files = {'file': uploaded_file}
        data = {
            'apikey': st.secrets["OCR_SPACE_API_KEY"],
            'language': 'eng',
            'isOverlayRequired': False
        }
        response = requests.post('https://api.ocr.space/parse/image',
                                 files=files,
                                 data=data)

        result = response.json()
        text = result["ParsedResults"][0]["ParsedText"]
        return text
    except Exception as e:
        return f"OCR failed via OCR.space: {e}"
        
def llm_extract_fields(text):
    print("\n[Agent 1] Extracting structured fields via LLM...")
    
    parser = PydanticOutputParser(pydantic_object=BankDetails)

    # 4. Create prompt template
    prompt = PromptTemplate(
        template=(
            "You are an expert financial document analyst.\n"
            "You will receive a raw text extracted from a financial document.\n"
            "Your task is to extract and structure the following banking details into clean, strictly formatted JSON:\n"
            "{format_instructions}\n\n"
            "Here is the raw extracted text:\n{text}"
        ),
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser
    try:
        result = chain.invoke({"text": text})
        logging.info(f"Extracted Data: {result}")
        return result
    except Exception as e:
        logging.error(f"LLM parsing failed: {e}")
        return {
            "account_holder_name": None,
            "account_holder_account_number": None,
            "account_holder_address": None,
            "routing_number": None,
            "bank_name": None,
            "bank_address": None,
            "error": "Could not parse extracted data. Please check document clarity or retry."
        }

# ----------- Agent 2: Validator (LLM-powered with enrichment trigger) -----------
def validate_user_input(user_input, extracted_data):
    print("\n[Agent 2] Validating fields via LLM...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent data validator assistant.
          You will receive:
          - Vendor-submitted banking details
          - Extracted banking details from the uploaded document

          Your task:
          1. Compare both data sets field by field:
            - account_holder_name
            - account_holder_account_number
            - account_holder_address
            - routing_number
            - bank_name
            - bank_address
          2. If a field is missing, null, or not found in either the vendor-submitted data or the extracted document data, set the result for that field to null — do not attempt to match or guess.
          3. If the values are not exactly the same but are semantically or partially matching (e.g., missing city name, formatting differences, or minor spelling errors), return **"Match"**.
          4. Only return **"Mismatch"** when the fields clearly refer to different information or entities.
          5. Output only structured JSON in this exact format — no explanations or extra text:

          {{
            "account_holder_name": "Match" | "Mismatch" | null,
            "account_holder_account_number": "Match" | "Mismatch" | null,
            "account_holder_address": "Match" | "Mismatch" | null,
            "routing_number": "Match" | "Mismatch" | null,
            "bank_name": "Match" | "Mismatch" | null,
            "bank_address": "Match" | "Mismatch" | null
          }}
          """),
        ("human", "Here is the vendor data:\n{user_input}\nHere is the extracted document data:\n{extracted_data}")
    ])
    chain = prompt | llm
    response = chain.invoke({
        "user_input": safe_json(user_input),
        "extracted_data": safe_json(extracted_data)
    })
    validation = response.content
    logging.info(f"Validation Results: {response.content}")
    return validation

# --------------- Global Account Number Validation ---------------
def is_valid_account_number(account_number: str):
    account_number_clean = account_number.replace(" ", "").replace("-", "")
    if re.match(r'^[A-Z]{2}\d{2}[A-Z0-9]{10,30}$', account_number_clean):
        return True  # IBAN-like
    if account_number_clean.isdigit() and 8 <= len(account_number_clean) <= 18:
        return True
    return False

# --------------- Bank Verification (Serper API) ---------------
def verify_bank_details_with_serper(bank_name, bank_address):
    try:
        headers = {
            'X-API-KEY': st.secrets["SERPER_API_KEY"],
            'Content-Type': 'application/json'
        }
        payload = {"q": f"{bank_name} {bank_address}", "gl": "us", "hl": "en"}
        response = requests.post('https://google.serper.dev/search', headers=headers, json=payload)
        results = response.json()
        for result in results.get('organic', []):
            title = result.get('title', '').lower()
            snippet = result.get('snippet', '').lower()
            if bank_name.lower() in title or bank_name.lower() in snippet:
                return True
        return False
    except Exception as e:
        print(f"SerperAPI Error: {e}")
        return False

# --------------- Forgery Detection (Image) ---------------
def detect_image_forgery(uploaded_file_path):
    try:
        img = cv2.imread(uploaded_file_path)
        if img is None:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < 100  # threshold
    except Exception as e:
        print(f"Forgery detection error: {e}")
        return False

# --------------- Fraud Detection Pipeline ---------------
def fraud_detection_pipeline(extracted_data):
    if isinstance(extracted_data, BaseModel):
        extracted_data = extracted_data.dict()
    elif isinstance(extracted_data, dict):
        extracted_data = extracted_data
    else:
        raise ValueError("Extracted data is not valid. Cannot proceed with fraud detection.")

    account_number = extracted_data.get("account_holder_account_number", "")
    bank_name = extracted_data.get("bank_name", "")
    bank_address = extracted_data.get("bank_address", "")

    print("\n🚀 Running Global Fraud Detection...")

    account_valid = is_valid_account_number(account_number)
    bank_verified = verify_bank_details_with_serper(bank_name, bank_address)
    # forgery_detected = detect_image_forgery(uploaded_file_path)

    fraud_score = 0
    if not account_valid:
        fraud_score += 30
    if not bank_verified:
        fraud_score += 40
    # if forgery_detected:
    #     fraud_score += 30

    fraud_score = min(fraud_score, 100)

    return {
        "account_number_valid": account_valid,
        "bank_verified": bank_verified,
        "forgery_detected": forgery_detected,
        "fraud_score": fraud_score
    }


# ----------- Streamlit UI -----------
st.title("🧠 LLM-Powered Banking Detail Validator")
st.caption("LangChain + OpenAI + OCR | Automated extraction and validation of bank document fields | Designed by Sushant Charaya")

with st.sidebar:
    st.header("📤 Upload Bank Document")
    uploaded_file = st.file_uploader("Upload Image or PDF", type=["png", "jpg", "jpeg"])

st.header("🏦 Enter Vendor Banking Details")
vendor_input = {
    "account_holder_name": st.text_input("Account Holder Name"),
    "account_holder_account_number": st.text_input("Account Number"),
    "account_holder_address": st.text_input("Account Holder Address"),
    "routing_number": st.text_input("Routing Number"),
    "bank_name": st.text_input("Bank Name"),
    "bank_address": st.text_input("Bank Address")
}

if st.button('✅ Validate Banking Details'):
    if uploaded_file and all(vendor_input.values()):
        st.info("Processing uploaded document...")
        extracted_text = extract_text_from_document(uploaded_file)
        st.info("Processing done ✅")

        # Run LLM extraction
        extracted_data = llm_extract_fields(extracted_text)
        st.subheader("📑 AI-Powered Extracted Fields")
        st.caption("Structured banking fields extracted by the LLM from the uploaded document.")
        if "error" in extracted_data:
            st.error(extracted_data["error"])
        else:
            st.json(extracted_data)

        # Run LLM validation
        validation_result = validate_user_input(vendor_input, extracted_data)
        st.subheader("🔍 AI Validation Output")
        st.caption("Comparison result using LLM between vendor-submitted data and document-extracted data.")
        st.code(validation_result, language="json")
        st.caption("""
        **🟢 Match** — The AI determined the vendor-submitted value and document-extracted value are the same or semantically similar (e.g., formatting or minor spelling differences).
        
        **🔴 Mismatch** — The values are clearly different or refer to unrelated entities.
        
        **⚠️ Null** — The field was missing, unreadable, or not found in the extracted document.
        """)
        
        fraud_report = fraud_detection_pipeline(safe_json(extracted_data))
        st.subheader("🚨 Fraud Detection Report")
        st.json(fraud_report)

        # if fraud_report["fraud_score"] >= 60:
        #     st.error("❗ High Fraud Risk Detected!")
        # elif fraud_report["fraud_score"] >= 30:
        #     st.warning("⚠️ Medium Risk - Manual Review Recommended.")
        # else:
        #     st.success("✅ Document Seems Safe.")
        
        st.markdown("---")
        st.success("""
        🚀 **Built with purpose by Sushant Charaya**
        
        This AI-powered tool leverages OCR, LLMs, and the LangChain framework to automate the validation of banking details — delivering unmatched speed, accuracy, and scalability.
        
        🧠 Designed for operational teams.  
        🔒 Future-ready: Fraud detection powered by AI module launching soon...
        """)
        
        logging.info("Fraud Detection complete")
    else:
        st.warning("Please upload a document and enter all vendor data to start.")
