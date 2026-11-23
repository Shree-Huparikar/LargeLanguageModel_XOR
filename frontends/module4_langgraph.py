"""Module 4 - Stateful LangGraph Workflow with SQLite Persistence & HITL."""
import uuid

import requests
import streamlit as st

from common import (
    MODULES,
    REQUEST_FIELDS,
    api_post,
    chips,
    empty_state,
    hero,
    pick,
    quote,
    setup_page,
    sidebar,
)

MODULE = "workflow"
setup_page(MODULE)
sidebar(MODULE)

st.session_state.setdefault("wf_messages", [])
st.session_state.setdefault("wf_thread_id", f"thread-{uuid.uuid4().hex[:8]}")
st.session_state.setdefault("wf_pending_hitl", None)
st.session_state.setdefault("wf_prev_thread_id", None)


def _api(path: str) -> str:
    return f"{st.session_state['backend_url'].rstrip('/')}{path}"


def _load_thread(thread_id: str) -> None:
    resp = requests.get(_api(f"/api/workflow/history/{thread_id}"), timeout=10)
    resp.raise_for_status()
    data = resp.json()
    st.session_state["wf_thread_id"] = thread_id
    st.session_state["wf_messages"] = data.get("messages", [])
    state = data.get("state") or {}
    st.session_state["wf_pending_hitl"] = {"thread_id": thread_id} if state.get("requires_approval") else None
    st.session_state["wf_last_snapshot"] = state


def _saved_threads() -> list[dict]:
    try:
        resp = requests.get(_api("/api/workflow/threads"), timeout=10)
        return resp.json().get("threads", []) if resp.ok else []
    except Exception:  # noqa: BLE001
        return []


if not st.session_state["wf_messages"] and st.session_state["wf_pending_hitl"] is None:
    try:
        _load_thread(st.session_state["wf_thread_id"])
    except Exception:  # noqa: BLE001
        pass

SUGGESTIONS = [
    "Where is my order #ORD-4821? It hasn't arrived yet.",
    "I want a refund of $89 for a damaged headphone order.",
    "What are your customer support working hours?",
]

threads = _saved_threads()
active_meta = next((t for t in threads if t["thread_id"] == st.session_state["wf_thread_id"]), None)
active_label = active_meta["title"] if active_meta else f"Thread {len(threads) + 1}"

with st.sidebar:
    st.markdown("**Try sample workflows**")
    for s in SUGGESTIONS:
        if st.button(s, key=f"wf_sugg_{s}"):
            st.session_state["wf_queued"] = s

    st.divider()
    st.markdown("**🧵 SQLite Threads**")
    st.caption(f"Active: **{active_label}**")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🆕 New Thread"):
            st.session_state["wf_prev_thread_id"] = st.session_state["wf_thread_id"]
            st.session_state["wf_thread_id"] = f"thread-{uuid.uuid4().hex[:8]}"
            st.session_state["wf_messages"] = []
            st.session_state["wf_pending_hitl"] = None
            st.session_state.pop("wf_last_snapshot", None)
            st.rerun()
    with c2:
        if st.button("🔍 State"):
            try:
                resp = requests.get(_api(f"/api/workflow/state/{st.session_state['wf_thread_id']}"), timeout=10)
                if resp.ok:
                    st.session_state["wf_last_snapshot"] = resp.json()
            except Exception as exc:  # noqa: BLE001
                st.error(f"State error: {exc}")

    for t in [x for x in threads if x["thread_id"] != st.session_state["wf_thread_id"]][:8]:
        label = t.get("title") or f"Thread {t.get('thread_num', '')}"
        cat_tag = f" · {t['last_category']}" if t.get("last_category") else ""
        if st.button(f"💬 {label} ({t['message_count']}){cat_tag}", key=f"th_{t['thread_id']}"):
            _load_thread(t["thread_id"])
            st.session_state["wf_prev_thread_id"] = None
            st.rerun()
        if t.get("last_question"):
            st.caption(f"↳ _{t['last_question'][:55]}_")

    if st.session_state.get("wf_last_snapshot"):
        with st.expander("Checkpointer Snapshot", expanded=True):
            st.json(st.session_state["wf_last_snapshot"])

hero(
    "LangGraph Stateful Workflow",
    "Conditional routing (Order / Refund / General), SQLite persistence, and Human-in-the-Loop approval.",
    MODULES[MODULE]["icon"],
)

prev_id = st.session_state.get("wf_prev_thread_id")
if prev_id and not st.session_state["wf_messages"]:
    prev_info = next((t for t in threads if t["thread_id"] == prev_id), None)
    next_num = max((t.get("thread_num", 0) for t in threads), default=0) + 1
    msg = f"**Thread {next_num}** is ready."
    if prev_info:
        prev_label = prev_info.get("title") or "Previous thread"
        msg += f" **{prev_label}** has **{prev_info['message_count']}** msgs"
        if prev_info.get("last_question"):
            msg += f" — last query: *\"{prev_info['last_question']}\"*."
    b_col, btn_col = st.columns([4, 1])
    with b_col:
        st.info(msg)
    with btn_col:
        if st.button("↩️ Reopen", key="reopen_prev"):
            _load_thread(prev_id)
            st.session_state["wf_prev_thread_id"] = None
            st.rerun()


def render_message(msg: dict) -> None:
    if msg["role"] == "user":
        st.markdown(msg["content"])
        return

    tags = [f"🧭 Route: {msg['category'].upper()}"] if msg.get("category") else []
    if msg.get("turn_count"):
        tags.append(f"🔄 Checkpoint Turn #{msg['turn_count']}")
    chips(tags)

    if msg.get("requires_approval"):
        st.warning("⏸️ Workflow Paused — Human-in-the-Loop Approval Required before executing refund.")
        if msg.get("draft_response"):
            quote(msg["draft_response"])
    else:
        st.markdown(msg.get("response") or msg.get("content") or "_No response returned._")

    if msg.get("steps_taken"):
        with st.expander("🗺️ Execution Trace (Nodes & Edges)", expanded=False):
            st.markdown(" → ".join(f"`{s}`" for s in msg["steps_taken"]))

    if msg.get("latency"):
        st.caption(f"Executed in {msg['latency']:.1f}s")


if not st.session_state["wf_messages"]:
    empty_state("🔀", "Send a support message or reopen a previous thread from SQLite.")

for m in st.session_state["wf_messages"]:
    with st.chat_message(m["role"], avatar="🧑‍💻" if m["role"] == "user" else "🔀"):
        render_message(m)

pending = st.session_state.get("wf_pending_hitl")
if pending:
    st.info(f"**Human-in-the-Loop Gate** — **{active_label}** is paused waiting for supervisor decision.")
    reviewer_note = st.text_input("Supervisor note (optional)", placeholder="e.g., Verified return; approved.", key="hitl_note_input")
    c1, c2 = st.columns(2)
    approve_clicked = c1.button("✅ Approve & Resume Graph", type="primary", use_container_width=True)
    reject_clicked = c2.button("❌ Reject Refund", use_container_width=True)

    if approve_clicked or reject_clicked:
        with st.spinner("Resuming checkpointed graph…"):
            data, secs, err = api_post(
                "/api/workflow/resume",
                {"thread_id": st.session_state["wf_thread_id"], "approved": bool(approve_clicked), "reviewer_note": reviewer_note},
            )
        if err:
            st.error(err)
        else:
            st.session_state["wf_pending_hitl"] = None
            st.session_state["wf_messages"].append(
                {
                    "role": "assistant",
                    "content": pick(data, "response"),
                    "category": pick(data, "category"),
                    "response": pick(data, "response"),
                    "draft_response": pick(data, "draft_response"),
                    "requires_approval": False,
                    "steps_taken": data.get("steps_taken", []),
                    "turn_count": data.get("turn_count", 0),
                    "latency": secs,
                }
            )
            st.rerun()

prompt = st.chat_input("Enter a support message (or resolve pending approval above)…", disabled=bool(pending))
if not prompt and st.session_state.get("wf_queued") and not pending:
    prompt = st.session_state.pop("wf_queued")

if prompt:
    user_msg = {"role": "user", "content": prompt}
    st.session_state["wf_messages"].append(user_msg)
    st.session_state["wf_prev_thread_id"] = None
    with st.chat_message("user", avatar="🧑‍💻"):
        render_message(user_msg)

    with st.chat_message("assistant", avatar="🔀"):
        with st.spinner("Running LangGraph workflow…"):
            data, secs, err = api_post(
                MODULES[MODULE]["endpoint"],
                {REQUEST_FIELDS["workflow"]: prompt, "thread_id": st.session_state["wf_thread_id"]},
            )
        if err:
            st.error(err)
            st.session_state["wf_messages"].pop()
        else:
            requires_approval = bool(data.get("requires_approval"))
            reply_text = pick(data, "response")
            reply = {
                "role": "assistant",
                "content": reply_text,
                "category": pick(data, "category"),
                "response": reply_text,
                "draft_response": pick(data, "draft_response"),
                "requires_approval": requires_approval,
                "steps_taken": data.get("steps_taken", []),
                "turn_count": data.get("turn_count", 0),
                "latency": secs,
            }
            st.session_state["wf_messages"].append(reply)
            if requires_approval:
                st.session_state["wf_pending_hitl"] = {"thread_id": st.session_state["wf_thread_id"]}
                st.rerun()
            else:
                render_message(reply)