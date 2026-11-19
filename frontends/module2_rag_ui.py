import requests
import streamlit as st

BASE_URL = "http://127.0.0.1:8000"
RAG_API_URL = f"{BASE_URL}/api/rag"
UPLOAD_API_URL = f"{BASE_URL}/api/rag/upload"


def extract_text_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text") is not None:
                parts.append(str(block["text"]))
        return "".join(parts).strip()
    if isinstance(content, dict):
        return str(content.get("text", "")).strip()
    return str(content).strip()


st.set_page_config(page_title="Module 2 - Document RAG", layout="centered")
st.title("Module 2: Document RAG")

uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_pdf is not None:
    upload_key = f"{uploaded_pdf.name}_{uploaded_pdf.size}"
    if st.session_state.get("last_uploaded_pdf") != upload_key:
        with st.spinner("Uploading and indexing PDF..."):
            try:
                files = {
                    "file": (
                        uploaded_pdf.name,
                        uploaded_pdf.getvalue(),
                        "application/pdf",
                    )
                }
                up_resp = requests.post(UPLOAD_API_URL, files=files, timeout=120)
                if not up_resp.ok:
                    st.error(f"Upload failed ({up_resp.status_code}): {up_resp.text}")
                up_resp.raise_for_status()
                st.session_state["last_uploaded_pdf"] = upload_key
                st.success(f"Uploaded: {uploaded_pdf.name}")
            except requests.RequestException as exc:
                st.error(f"Upload request failed: {exc}")

if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []

for msg in st.session_state.rag_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question from your PDF...")

if prompt:
    st.session_state.rag_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching PDF..."):
            try:
                resp = requests.post(RAG_API_URL, json={"question": prompt}, timeout=60)
                if not resp.ok:
                    st.error(f"Backend error ({resp.status_code}): {resp.text}")
                resp.raise_for_status()

                data = resp.json()
                raw_reply = (
                    data.get("answer")
                    or data.get("response")
                    or data.get("result")
                    or data.get("ai_history_entry")
                    or ""
                )
                clean_reply = extract_text_content(raw_reply)

                st.markdown(clean_reply)
                st.session_state.rag_messages.append(
                    {"role": "assistant", "content": clean_reply}
                )
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")