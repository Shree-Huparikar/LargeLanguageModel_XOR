import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/api/chat"


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


def format_chat_reply(data: dict) -> str:
    topic = extract_text_content(data.get("topic"))
    answer = extract_text_content(
        data.get("answer")
        or data.get("response")
        or data.get("result")
        or ""
    )
    example = extract_text_content(data.get("example"))

    sections = []
    if topic:
        sections.append(f"**Topic:** {topic}")
    if answer:
        sections.append(f"**Answer:** {answer}")
    if example:
        sections.append(f"**Example:** {example}")

    if sections:
        return "\n\n".join(sections)

    return extract_text_content(data.get("ai_history_entry"))


st.set_page_config(page_title="Module 1 - Structured Chat", layout="centered")
st.title("Module 1: Structured Chat")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question...")

if prompt:
    history_turns = [
        {
            "role": m["role"],
            "content": extract_text_content(m.get("raw_content", m["content"])),
        }
        for m in st.session_state.chat_messages
    ]

    st.session_state.chat_messages.append(
        {"role": "user", "content": prompt, "raw_content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {"question": prompt, "history": history_turns}
                resp = requests.post(API_URL, json=payload, timeout=60)
                resp.raise_for_status()

                data = resp.json()
                display_reply = format_chat_reply(data)
                history_entry = (
                    extract_text_content(data.get("ai_history_entry"))
                    or display_reply
                )

                st.markdown(display_reply)
                st.session_state.chat_messages.append(
                    {
                        "role": "assistant",
                        "content": display_reply,
                        "raw_content": history_entry,
                    }
                )
            except requests.HTTPError as exc:
                code = exc.response.status_code if exc.response is not None else "error"
                detail = exc.response.text if exc.response is not None else str(exc)
                st.error(f"Backend error ({code}): {detail}")
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")