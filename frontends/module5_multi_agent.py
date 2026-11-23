"""Module 5 - Multi-Agent Travel Planning System (Supervisor + Specialist Agents)."""
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

MODULE = "multi_agent"
setup_page(MODULE)
sidebar(MODULE)

st.session_state.setdefault("ma_history", [])

SUGGESTIONS = [
    "Plan a 4-day budget trip to Jaipur from Mumbai for 2 people under ₹25,000",
    "5-day solo backpacking trip to Manali with adventure activities under ₹18,000",
    "3-day weekend family trip to Goa with beach hotels and sightseeing under ₹40,000",
]

AGENT_LABELS = {
    "travel": "🗺️ Travel Agent",
    "budget": "💰 Budget Agent",
    "booking": "🏨 Booking Agent",
    "activities": "🎯 Activities Agent",
}

with st.sidebar:
    st.markdown("**Try sample trip requests**")
    for s in SUGGESTIONS:
        if st.button(s, key=f"ma_sugg_{s}"):
            st.session_state["ma_queued"] = s
    st.divider()
    if st.button("🗑️ Clear history"):
        st.session_state["ma_history"] = []
        st.rerun()

hero(
    "Multi-Agent Travel Planner",
    "A Supervisor coordinator delegates to Travel, Budget, Booking, and Activities specialists via shared state and handoffs.",
    MODULES[MODULE]["icon"],
)


def render_entry(entry: dict) -> None:
    if entry["role"] == "user":
        st.markdown(entry["content"])
        return

    agents = entry.get("agents_used") or []
    chips([AGENT_LABELS.get(a, a) for a in agents], style="green")

    # Unified Final Plan
    st.markdown(entry.get("result") or "_No travel plan returned._")

    # Individual Specialist Outputs (Shared Context inspection)
    tabs_available = [
        ("🗺️ Travel", entry.get("travel")),
        ("💰 Budget", entry.get("budget")),
        ("🏨 Booking", entry.get("booking")),
        ("🎯 Activities", entry.get("activities")),
    ]
    non_empty = [(label, body) for label, body in tabs_available if body]
    if non_empty:
        with st.expander("🔍 Inspect Individual Specialist Outputs (Shared State)", expanded=False):
            tabs = st.tabs([label for label, _ in non_empty])
            for tab, (_, body) in zip(tabs, non_empty):
                with tab:
                    st.markdown(body)

    # Supervisor Delegation & Agent Handoff Trace
    handoffs = entry.get("handoff_log") or []
    if handoffs:
        with st.expander("🤝 Supervisor Delegation & Handoff Log", expanded=False):
            for idx, step in enumerate(handoffs, start=1):
                st.markdown(f"{idx}. `{step}`")

    if entry.get("latency"):
        st.caption(f"Collaborated & synthesized in {entry['latency']:.1f}s")


if not st.session_state["ma_history"]:
    empty_state(
        "🤝",
        "Enter your destination, trip duration, and budget below—or pick a sample trip from the sidebar.",
    )

for item in st.session_state["ma_history"]:
    with st.chat_message(item["role"], avatar="🧑‍💻" if item["role"] == "user" else "🤝"):
        render_entry(item)

prompt = st.chat_input("Enter destination, duration, and budget…")
if not prompt and st.session_state.get("ma_queued"):
    prompt = st.session_state.pop("ma_queued")

if prompt:
    user_msg = {"role": "user", "content": prompt}
    st.session_state["ma_history"].append(user_msg)
    with st.chat_message("user", avatar="🧑‍💻"):
        render_entry(user_msg)

    with st.chat_message("assistant", avatar="🤝"):
        with st.spinner("Supervisor coordinating Travel, Budget, Booking & Activities agents…"):
            data, secs, err = api_post(
                MODULES[MODULE]["endpoint"],
                {REQUEST_FIELDS["multi_agent"]: prompt},
            )
        if err:
            st.error(err)
            st.session_state["ma_history"].pop()
        else:
            reply = {
                "role": "assistant",
                "result": pick(data, "result", default="_No travel plan returned._"),
                "travel": pick(data, "travel"),
                "budget": pick(data, "budget"),
                "booking": pick(data, "booking"),
                "activities": pick(data, "activities"),
                "agents_used": data.get("agents_used", []),
                "handoff_log": data.get("handoff_log", []),
                "latency": secs,
            }
            st.session_state["ma_history"].append(reply)
            render_entry(reply)