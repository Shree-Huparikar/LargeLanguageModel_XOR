"""Module 3 - AI agent with tool calling."""
import streamlit as st

from common import (REQUEST_FIELDS, MODULES, api_post, chips, empty_state,
                    hero, pick, setup_page, sidebar)

MODULE = "tools"
setup_page(MODULE)
sidebar(MODULE)

st.session_state.setdefault("tool_messages", [])

TOOLS = [
    ("➕", "add_numbers", "Add two numbers", "What is 245 + 378?"),
    ("✖️", "multiply_numbers", "Multiply two numbers", "Multiply 12 by 15"),
    ("📅", "get_current_date", "Get today's date", "What is today's date?"),
    ("🎓", "get_student_info", "Look up a student", "Get the student information"),
]

with st.sidebar:
    st.markdown("**Available tools**")
    for icon, name, desc, example in TOOLS:
        if st.button(f"{icon} {desc}", key=f"tool_{name}", help=f"Runs `{name}`"):
            st.session_state["queued"] = example
    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state["tool_messages"] = []
        st.rerun()

hero("Tool-Calling Agent", "The agent decides which tool to use, runs it, and explains the result.", MODULES[MODULE]["icon"])


def render(msg: dict) -> None:
    if msg["role"] == "user":
        st.markdown(msg["content"])
        return
    st.markdown(msg["answer"])
    if msg.get("tools"):
        chips([f"🛠 {t}" for t in msg["tools"]], "amber")
    if msg.get("latency"):
        st.caption(f"Responded in {msg['latency']:.1f}s")


def tool_names(data) -> list:
    raw = pick(data, "tools_used", "tool_calls", "tools", default=[])
    if isinstance(raw, str):
        return [raw]
    out = []
    for t in raw or []:
        out.append(t.get("name", str(t)) if isinstance(t, dict) else str(t))
    return out


if not st.session_state["tool_messages"]:
    empty_state("🛠️", "Ask a question that needs a tool, or pick one from the sidebar.")

for m in st.session_state["tool_messages"]:
    with st.chat_message(m["role"], avatar="🧑‍💻" if m["role"] == "user" else "🤖"):
        render(m)

prompt = st.chat_input("Ask the agent to calculate, check the date, or look up a student…")
if not prompt and st.session_state.get("queued"):
    prompt = st.session_state.pop("queued")

if prompt:
    user_msg = {"role": "user", "content": prompt}
    st.session_state["tool_messages"].append(user_msg)
    with st.chat_message("user", avatar="🧑‍💻"):
        render(user_msg)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Running agent…"):
            data, secs, err = api_post(MODULES[MODULE]["endpoint"], {REQUEST_FIELDS["tools"]: prompt})
        if err:
            st.error(err)
            st.session_state["tool_messages"].pop()
        else:
            reply = {
                "role": "assistant",
                "answer": pick(data, "answer", "response", "output", "result", default="_No answer returned._"),
                "tools": tool_names(data),
                "latency": secs,
            }
            st.session_state["tool_messages"].append(reply)
            render(reply)
