"""Module 1 - Chat assistant with structured output (answer / topic / example)."""
import streamlit as st

from common import (
    MODULES,
    REQUEST_FIELDS,
    api_post,
    chips,
    empty_state,
    hero,
    pick,
    setup_page,
    sidebar,
)

MODULE = "chat"
setup_page(MODULE)
sidebar(MODULE)

st.session_state.setdefault("chat_messages", [])

SUGGESTIONS = [
    "Explain what a vector database is",
    "What is the difference between RAG and fine-tuning?",
    "How do prompt templates work in LangChain?",
]


def build_history(messages: list[dict]) -> list[dict[str, str]]:
    """Format prior UI messages into `ChatTurn` dicts expected by `ChatRequest.history`."""
    turns: list[dict[str, str]] = []
    for m in messages:
        role = m.get("role")
        if role == "user":
            content = (m.get("content") or "").strip()
            if content:
                turns.append({"role": "user", "content": content})
        elif role == "assistant":
            entry = (m.get("ai_history_entry") or "").strip()
            if not entry:
                answer = (m.get("answer") or "").strip()
                if answer and answer != "_No answer returned._":
                    topic = (m.get("topic") or "General").strip()
                    example = (m.get("example") or "N/A").strip()
                    entry = f"Topic: {topic}\nAnswer: {answer}\nExample: {example}"
            if entry:
                turns.append({"role": "assistant", "content": entry})
    return turns


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

hero(
    "Chat Assistant",
    "Ask anything. Answers come back structured with a topic and an example.",
    MODULES[MODULE]["icon"],
)


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
    empty_state(
        "💬",
        "Start a conversation using the box below or pick a suggestion from the sidebar.",
    )

for m in st.session_state["chat_messages"]:
    with st.chat_message(m["role"], avatar="🧑‍💻" if m["role"] == "user" else "🤖"):
        render(m)

prompt = st.chat_input("Type your question…")
if not prompt and st.session_state.get("queued"):
    prompt = st.session_state.pop("queued")

if prompt:
    prior_history = build_history(st.session_state["chat_messages"])

    user_msg = {"role": "user", "content": prompt}
    st.session_state["chat_messages"].append(user_msg)
    with st.chat_message("user", avatar="🧑‍💻"):
        render(user_msg)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            data, secs, err = api_post(
                MODULES[MODULE]["endpoint"],
                {
                    REQUEST_FIELDS["chat"]: prompt,
                    "history": prior_history,
                },
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
                "ai_history_entry": pick(data, "ai_history_entry"),
                "latency": secs,
            }
            st.session_state["chat_messages"].append(reply)
            render(reply)