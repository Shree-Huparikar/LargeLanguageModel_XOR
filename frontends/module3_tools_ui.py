import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/api/tools"


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


st.set_page_config(page_title="Module 3 - AI Agent Tools", layout="centered")
st.title("Module 3: AI Agent Tools")

if "tool_messages" not in st.session_state:
    st.session_state.tool_messages = []

for msg in st.session_state.tool_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask the agent to use a tool (e.g., Multiply 3 and 5)...")

if prompt:
    st.session_state.tool_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Running agent..."):
            try:
                resp = requests.post(API_URL, json={"prompt": prompt}, timeout=60)
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
                st.session_state.tool_messages.append(
                    {"role": "assistant", "content": clean_reply}
                )
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")