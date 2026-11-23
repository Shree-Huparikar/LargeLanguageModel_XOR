"""Module 4 - Stateful LangGraph Workflow with Conditional Routing, Checkpointing & HITL."""
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

SUGGESTIONS = [
    "Where is my order #ORD-4821? It hasn't arrived yet.",
    "I want a refund of $89 for a damaged headphone order.",
    "What are your customer support working hours?",
]

with st.sidebar:
    st.markdown("**Try sample workflows**")
    for s in SUGGESTIONS:
        if st.button(s, key=f"wf_sugg_{s}"):
            st.session_state["wf_queued"] = s

    st.divider()
    st.markdown("**🧵 Checkpoint Persistence**")
    st.caption(f"Thread ID: `{st.session_state['wf_thread_id']}`")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🆕 New Thread"):
            st.session_state["wf_thread_id"] = f"thread-{uuid.uuid4().hex[:8]}"
            st.session_state["wf_messages"] = []
            st.session_state["wf_pending_hitl"] = None
            st.rerun()
    with col_b:
        if st.button("🔍 Inspect State"):
            try:
                resp = requests.get(
                    f"{st.session_state['backend_url']}/api/workflow/state/{st.session_state['wf_thread_id']}",
                    timeout=10,
                )
                if resp.ok:
                    st.session_state["wf_last_snapshot"] = resp.json()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not fetch state: {exc}")

    if st.session_state.get("wf_last_snapshot"):
        with st.expander("Checkpointer Snapshot", expanded=True):
            st.json(st.session_state["wf_last_snapshot"])

hero(
    "LangGraph Stateful Workflow",
    "Conditional routing (Order / Refund / General), MemorySaver checkpointing, and Human-in-the-Loop approval.",
    MODULES[MODULE]["icon"],
)


def render_message(msg: dict) -> None:
    if msg["role"] == "user":
        st.markdown(msg["content"])
        return

    cat = msg.get("category", "")
    steps = msg.get("steps_taken") or []
    turn = msg.get("turn_count", 0)

    tag_list = []
    if cat:
        tag_list.append(f"🧭 Route: {cat.upper()}")
    if turn:
        tag_list.append(f"🔄 Checkpoint Turn #{turn}")
    chips(tag_list)

    if msg.get("requires_approval"):
        st.warning("⏸️ Workflow Paused — Human-in-the-Loop Approval Required before executing refund.")
        if msg.get("draft_response"):
            quote(msg["draft_response"])
    else:
        st.markdown(msg.get("response") or "_No response returned._")

    if steps:
        with st.expander("🗺️ Execution Trace (Nodes & Edges)", expanded=False):
            st.markdown(" → ".join(f"`{s}`" for s in steps))

    if msg.get("latency"):
        st.caption(f"Executed in {msg['latency']:.1f}s")


if not st.session_state["wf_messages"]:
    empty_state(
        "🔀",
        "Send an order, refund, or general support message below. Refund requests trigger a Human-in-the-Loop approval checkpoint.",
    )

for m in st.session_state["wf_messages"]:
    with st.chat_message(m["role"], avatar="🧑‍💻" if m["role"] == "user" else "🔀"):
        render_message(m)

# --------------------------------------------------------------------------
# Human-in-the-Loop Approval Banner (when graph is interrupted)
# --------------------------------------------------------------------------
pending = st.session_state.get("wf_pending_hitl")
if pending:
    st.info(
        f"**Human-in-the-Loop Gate (`interrupt_before=['refund_execute']`)** — "
        f"Thread `{pending['thread_id']}` is paused waiting for supervisor decision."
    )
    reviewer_note = st.text_input(
        "Supervisor note (optional)",
        placeholder="e.g., Verified return tracking #RT-992; approved.",
        key="hitl_note_input",
    )
    c1, c2 = st.columns(2)
    with c1:
        approve_clicked = st.button("✅ Approve & Resume Graph", type="primary", use_container_width=True)
    with c2:
        reject_clicked = st.button("❌ Reject Refund", use_container_width=True)

    if approve_clicked or reject_clicked:
        decision = bool(approve_clicked)
        with st.spinner("Resuming checkpointed graph…"):
            data, secs, err = api_post(
                "/api/workflow/resume",
                {
                    "thread_id": st.session_state["wf_thread_id"],
                    "approved": decision,
                    "reviewer_note": reviewer_note,
                },
            )
        if err:
            st.error(err)
        else:
            st.session_state["wf_pending_hitl"] = None
            st.session_state["wf_messages"].append(
                {
                    "role": "assistant",
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

# --------------------------------------------------------------------------
# Chat Input
# --------------------------------------------------------------------------
prompt = st.chat_input(
    "Enter a support message (or resolve the pending approval above)…",
    disabled=bool(pending),
)
if not prompt and st.session_state.get("wf_queued") and not pending:
    prompt = st.session_state.pop("wf_queued")

if prompt:
    user_msg = {"role": "user", "content": prompt}
    st.session_state["wf_messages"].append(user_msg)
    with st.chat_message("user", avatar="🧑‍💻"):
        render_message(user_msg)

    with st.chat_message("assistant", avatar="🔀"):
        with st.spinner("Running LangGraph workflow…"):
            data, secs, err = api_post(
                MODULES[MODULE]["endpoint"],
                {
                    REQUEST_FIELDS["workflow"]: prompt,
                    "thread_id": st.session_state["wf_thread_id"],
                },
            )
        if err:
            st.error(err)
            st.session_state["wf_messages"].pop()
        else:
            requires_approval = bool(data.get("requires_approval"))
            reply = {
                "role": "assistant",
                "category": pick(data, "category"),
                "response": pick(data, "response"),
                "draft_response": pick(data, "draft_response"),
                "requires_approval": requires_approval,
                "steps_taken": data.get("steps_taken", []),
                "turn_count": data.get("turn_count", 0),
                "latency": secs,
            }
            st.session_state["wf_messages"].append(reply)
            if requires_approval:
                st.session_state["wf_pending_hitl"] = {
                    "thread_id": st.session_state["wf_thread_id"]
                }
                st.rerun()
            else:
                render_message(reply)