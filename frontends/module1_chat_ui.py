"""Module 1 - Chat assistant with structured output (answer / topic / example)."""
import streamlit as st

from common import (REQUEST_FIELDS, MODULES, api_post, chips, empty_state,
                    hero, pick, setup_page, sidebar)

MODULE = "chat"
setup_page(MODULE)
sidebar(MODULE)

st.session_state.setdefault("chat_messages", [])

SUGGESTIONS = [
    "Explain what a vector database is",
    "What is the difference between RAG and fine-tuning?",
    "How do prompt templates work in LangChain?",
]

with st.sidebar:
    st.markdown("**Try asking**")
    for s in SUGGESTIONS:
        if st.button(s, key=f"sugg_{s}"):
            st.session_state["queued"] = s
    st.divider()
    n = len(st.session_state["chat_messages"])
    st.caption(f"{n} message{'s' if n != 1 else ''} in this conversation")
    if st.button("🗑️ Clear conversation"):
        st.session_state["chat_messages"] = []
        st.rerun()

hero("Chat Assistant", "Ask anything. Answers come back structured with a topic and an example.", MODULES[MODULE]["icon"])


def render(msg: dict) -> None:
    if msg["role"] == "user":
        st.markdown(msg["content"])
        return
    st.markdown(msg["answer"])
    chips([f"🏷 {msg['topic']}"] if msg.get("topic") else [])
    if msg.get("example"):
        with st.expander("💡 Example", expanded=True):
            st.markdown(msg["example"])
    if msg.get("latency"):
        st.caption(f"Responded in {msg['latency']:.1f}s")


if not st.session_state["chat_messages"]:
    empty_state("💬", "Start a conversation using the box below or pick a suggestion from the sidebar.")

for m in st.session_state["chat_messages"]:
    with st.chat_message(m["role"], avatar="🧑‍💻" if m["role"] == "user" else "🤖"):
        render(m)

prompt = st.chat_input("Type your question…")
if not prompt and st.session_state.get("queued"):
    prompt = st.session_state.pop("queued")

if prompt:
    user_msg = {"role": "user", "content": prompt}
    st.session_state["chat_messages"].append(user_msg)
    with st.chat_message("user", avatar="🧑‍💻"):
        render(user_msg)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            data, secs, err = api_post(
                MODULES[MODULE]["endpoint"],
                {REQUEST_FIELDS["chat"]: prompt, "session_id": st.session_state["session_id"]},
            )
        if err:
            st.error(err)
            st.session_state["chat_messages"].pop()
        else:
            reply = {
                "role": "assistant",
                "answer": pick(data, "answer", "response", "output", default="_No answer returned._"),
                "topic": pick(data, "topic"),
                "example": pick(data, "example"),
                "latency": secs,
            }
            st.session_state["chat_messages"].append(reply)
            render(reply)
