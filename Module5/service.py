"""Module 5 - Multi-Agent Systems: Supervisor + Specialist Agents with Shared Context & Handoffs."""
import os
from pathlib import Path
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


# --------------------------------------------------------------------------
# 1. Shared Multi-Agent Context (State) & Supervisor Schema
# --------------------------------------------------------------------------
class TravelState(TypedDict):
    request: str
    next_agent: str
    supervisor_reason: str
    travel: str
    budget: str
    booking: str
    activities: str
    result: str
    handoff_log: list[str]
    iterations: int


class SupervisorDecision(BaseModel):
    next_agent: Literal["travel", "budget", "booking", "activities", "final"] = Field(
        description="Next specialist agent to delegate to, or 'final' when ready to synthesize."
    )
    reason: str = Field(
        description="Short explanation of why this specialist is being delegated to next."
    )


def _get_llm(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)


def _extract_text(content: Any) -> str:
    """Normalize LangChain/Gemini message content (str or list of blocks) to plain str."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("content"), str):
                    parts.append(block["content"])
            else:
                text_attr = getattr(block, "text", None)
                if isinstance(text_attr, str):
                    parts.append(text_attr)
        return "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    if content is None:
        return ""
    return str(content).strip()


# --------------------------------------------------------------------------
# 2. Supervisor Coordinator Node (Delegation & Handoff Orchestrator)
# --------------------------------------------------------------------------
ALL_SPECIALISTS = ["travel", "budget", "booking", "activities"]


def supervisor_node(state: TravelState) -> dict:
    """Coordinator agent that inspects shared state and delegates to the next specialist."""
    iterations = (state.get("iterations") or 0) + 1
    completed = [name for name in ALL_SPECIALISTS if (state.get(name) or "").strip()]
    remaining = [name for name in ALL_SPECIALISTS if name not in completed]
    log = list(state.get("handoff_log") or [])

    # Guardrail: once all 4 specialists have contributed or max hops reached, synthesize
    if not remaining or iterations > 6:
        reason = "All specialist inputs collected; delegating to Final Synthesizer."
        log.append(f"Supervisor -> final ({reason})")
        return {
            "next_agent": "final",
            "supervisor_reason": reason,
            "handoff_log": log,
            "iterations": iterations,
        }

    next_choice: str = remaining[0]
    reason = f"Delegating to {next_choice} specialist next."

    try:
        llm = _get_llm(temperature=0.0).with_structured_output(SupervisorDecision)
        prompt = (
            "You are the Travel Coordinator Supervisor managing 4 specialist agents:\n"
            "- 'travel': builds day-by-day route and itinerary\n"
            "- 'budget': estimates transport, stay, food, and activity costs\n"
            "- 'booking': recommends areas, hotel tiers, and booking tips (no fake reservations)\n"
            "- 'activities': curates local attractions, food spots, and experiences\n"
            "- 'final': combines all specialist outputs into the final user plan\n\n"
            f"User Request: {state['request']}\n"
            f"Already Completed Specialists: {completed or 'None'}\n"
            f"Remaining Specialists: {remaining}\n\n"
            "Pick the best next specialist from Remaining Specialists (or 'final' if the user "
            "explicitly asked ONLY for one narrow item and it is already completed)."
        )
        decision = llm.invoke(prompt)
        if decision is not None:
            candidate = decision.next_agent
            if candidate in remaining or (candidate == "final" and completed):
                next_choice = candidate
                reason = decision.reason
    except Exception:
        next_choice = remaining[0]

    log.append(f"Supervisor -> {next_choice}: {reason}")
    return {
        "next_agent": next_choice,
        "supervisor_reason": reason,
        "handoff_log": log,
        "iterations": iterations,
    }


# --------------------------------------------------------------------------
# 3. Specialist Agents (Read Shared Context -> Write Domain Output -> Handoff)
# --------------------------------------------------------------------------
def travel_agent(state: TravelState) -> dict:
    """Specialist 1: Day-by-day itinerary and logistics."""
    llm = _get_llm()
    reply = llm.invoke(
        "You are a Travel Itinerary Specialist. Create a clear, practical day-by-day travel plan "
        "(transport, route, and daily structure) for this request:\n\n"
        f"{state['request']}"
    )
    text = _extract_text(reply.content)
    log = list(state.get("handoff_log") or [])
    log.append("Travel Agent -> completed itinerary & handed off to Supervisor")
    return {"travel": text, "handoff_log": log}


def budget_agent(state: TravelState) -> dict:
    """Specialist 2: Cost breakdown informed by shared travel context."""
    llm = _get_llm()
    shared_travel = state.get("travel") or "Not yet drafted."
    reply = llm.invoke(
        "You are a Travel Budget Specialist. Estimate realistic travel, accommodation, food, "
        "and local transport costs in a clean breakdown.\n\n"
        f"User Request: {state['request']}\n"
        f"Shared Itinerary Context:\n{shared_travel}"
    )
    text = _extract_text(reply.content)
    log = list(state.get("handoff_log") or [])
    log.append("Budget Agent -> completed cost breakdown & handed off to Supervisor")
    return {"budget": text, "handoff_log": log}


def booking_agent(state: TravelState) -> dict:
    """Specialist 3: Hotel & stay recommendations aligned with shared budget/itinerary."""
    llm = _get_llm()
    shared_budget = state.get("budget") or "Not yet estimated."
    reply = llm.invoke(
        "You are a Hotel & Stay Booking Specialist. Suggest neighborhoods, affordable hotel tiers, "
        "and smart booking tips matching the request and budget. Do NOT claim to make actual reservations.\n\n"
        f"User Request: {state['request']}\n"
        f"Shared Budget Context:\n{shared_budget}"
    )
    text = _extract_text(reply.content)
    log = list(state.get("handoff_log") or [])
    log.append("Booking Agent -> completed stay recommendations & handed off to Supervisor")
    return {"booking": text, "handoff_log": log}


def activity_agent(state: TravelState) -> dict:
    """Specialist 4: Sightseeing, food, and local experiences."""
    llm = _get_llm()
    shared_travel = state.get("travel") or "Not yet drafted."
    reply = llm.invoke(
        "You are a Local Activities & Experiences Specialist. Suggest must-see tourist spots, "
        "local food experiences, and hidden gems aligned with the itinerary.\n\n"
        f"User Request: {state['request']}\n"
        f"Shared Itinerary Context:\n{shared_travel}"
    )
    text = _extract_text(reply.content)
    log = list(state.get("handoff_log") or [])
    log.append("Activities Agent -> completed recommendations & handed off to Supervisor")
    return {"activities": text, "handoff_log": log}


def final_synthesizer_agent(state: TravelState) -> dict:
    """Synthesizer: Merges all specialist outputs from shared state into one cohesive plan."""
    llm = _get_llm(temperature=0.2)
    reply = llm.invoke(
        "You are the Lead Travel Concierge. Combine the specialist reports below into a well-structured, "
        "easy-to-read final travel plan using clear Markdown sections. Omit any section marked 'N/A'.\n\n"
        f"Original Request: {state['request']}\n\n"
        f"--- Travel Plan ---\n{state.get('travel') or 'N/A'}\n\n"
        f"--- Budget Estimate ---\n{state.get('budget') or 'N/A'}\n\n"
        f"--- Hotels & Booking Tips ---\n{state.get('booking') or 'N/A'}\n\n"
        f"--- Activities & Experiences ---\n{state.get('activities') or 'N/A'}"
    )
    text = _extract_text(reply.content)
    log = list(state.get("handoff_log") or [])
    log.append("Final Synthesizer -> compiled unified travel plan")
    return {"result": text, "handoff_log": log}


# --------------------------------------------------------------------------
# 4. Routing & Graph Compilation
# --------------------------------------------------------------------------
def route_from_supervisor(
    state: TravelState,
) -> Literal["travel", "budget", "booking", "activities", "final"]:
    choice = state.get("next_agent", "final")
    if choice in ("travel", "budget", "booking", "activities", "final"):
        return choice
    return "final"


def build_module5_graph():
    graph = StateGraph(TravelState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("travel", travel_agent)
    graph.add_node("budget", budget_agent)
    graph.add_node("booking", booking_agent)
    graph.add_node("activities", activity_agent)
    graph.add_node("final", final_synthesizer_agent)

    # Entry -> Supervisor Coordinator
    graph.add_edge(START, "supervisor")

    # Supervisor delegates dynamically to specialists or final synthesizer
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "travel": "travel",
            "budget": "budget",
            "booking": "booking",
            "activities": "activities",
            "final": "final",
        },
    )

    # Every specialist hands back to the Supervisor for next-step orchestration
    graph.add_edge("travel", "supervisor")
    graph.add_edge("budget", "supervisor")
    graph.add_edge("booking", "supervisor")
    graph.add_edge("activities", "supervisor")

    graph.add_edge("final", END)

    return graph.compile()


# --------------------------------------------------------------------------
# 5. Public Execution Helper
# --------------------------------------------------------------------------
def run_multi_agent(request: str, graph=None) -> dict:
    if graph is None:
        graph = build_module5_graph()

    initial_state: TravelState = {
        "request": request,
        "next_agent": "",
        "supervisor_reason": "",
        "travel": "",
        "budget": "",
        "booking": "",
        "activities": "",
        "result": "",
        "handoff_log": [],
        "iterations": 0,
    }
    output = graph.invoke(initial_state)

    agents_used = [name for name in ALL_SPECIALISTS if (output.get(name) or "").strip()]
    return {
        "result": _extract_text(output.get("result", "")),
        "travel": _extract_text(output.get("travel", "")),
        "budget": _extract_text(output.get("budget", "")),
        "booking": _extract_text(output.get("booking", "")),
        "activities": _extract_text(output.get("activities", "")),
        "agents_used": agents_used,
        "handoff_log": output.get("handoff_log", []),
    }