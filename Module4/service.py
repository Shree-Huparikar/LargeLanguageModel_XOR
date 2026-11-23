"""Module 4 - LangGraph Stateful Support Workflow with Checkpointing & Human-in-the-Loop."""
import os
from typing import Any, Literal, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# 1. State & Structured Classification Schema
# --------------------------------------------------------------------------
class SupportState(TypedDict):
    question: str
    category: str
    requires_approval: bool
    approved: bool | None
    reviewer_note: str
    draft_response: str
    final_response: str
    steps_taken: list[str]
    turn_count: int


class RouteDecision(BaseModel):
    category: Literal["order", "refund", "general"] = Field(
        description="Classify user query into 'order', 'refund', or 'general'."
    )
    requires_approval: bool = Field(
        description="True ONLY if category is 'refund' and user asks for a refund/money back."
    )


def _get_llm() -> ChatGoogleGenerativeAI:
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    return ChatGoogleGenerativeAI(model=model_name, temperature=0.2)


def _extract_text(content: Any) -> str:
    """Normalize LangChain/Gemini message content (str or list of blocks) to a plain string."""
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
# 2. Graph Nodes
# --------------------------------------------------------------------------
def classify_node(state: SupportState) -> dict:
    """Classify incoming support query using structured Gemini output."""
    llm = _get_llm().with_structured_output(RouteDecision)
    prompt = (
        "You are a customer support triage router.\n"
        "Classify the user's message into one of: 'order', 'refund', or 'general'.\n"
        "Set requires_approval=True whenever category is 'refund'.\n\n"
        f"User message: {state['question']}"
    )
    decision: RouteDecision = llm.invoke(prompt)
    steps = list(state.get("steps_taken") or [])
    steps.append(f"classify -> {decision.category}")
    return {
        "category": decision.category,
        "requires_approval": decision.requires_approval,
        "approved": None,
        "reviewer_note": "",
        "steps_taken": steps,
        "turn_count": (state.get("turn_count") or 0) + 1,
    }


def order_node(state: SupportState) -> dict:
    """Handle order tracking, shipping, and delivery inquiries."""
    llm = _get_llm()
    reply = llm.invoke(
        "You are an Order Support Specialist. Give a helpful, concise response "
        "explaining how to check order status/tracking and next steps.\n\n"
        f"Customer query: {state['question']}"
    )
    text = _extract_text(reply.content)
    steps = list(state.get("steps_taken") or [])
    steps.append("order_node")
    return {
        "draft_response": text,
        "final_response": text,
        "steps_taken": steps,
    }


def refund_prepare_node(state: SupportState) -> dict:
    """Prepare refund assessment before pausing for Human-in-the-Loop approval."""
    llm = _get_llm()
    draft = llm.invoke(
        "You are a Refund Specialist drafting a proposed refund resolution for manager review. "
        "Summarize the refund request and proposed action in 2-3 sentences.\n\n"
        f"Customer query: {state['question']}"
    )
    draft_text = _extract_text(draft.content)
    steps = list(state.get("steps_taken") or [])
    steps.append("refund_prepare (awaiting human approval)")
    return {
        "draft_response": draft_text,
        "steps_taken": steps,
    }


def refund_execute_node(state: SupportState) -> dict:
    """Executed AFTER human approval/rejection checkpoint."""
    steps = list(state.get("steps_taken") or [])
    note = (state.get("reviewer_note") or "").strip()
    draft_text = _extract_text(state.get("draft_response", ""))

    if state.get("approved"):
        steps.append("refund_execute (APPROVED)")
        msg = (
            f"**Refund Approved**\n\n{draft_text}\n\n"
            "Your refund has been authorized by a support supervisor and will reflect within 3-5 business days."
        )
        if note:
            msg += f"\n\n*Supervisor note:* {note}"
    else:
        steps.append("refund_execute (REJECTED)")
        msg = (
            "**Refund Request Declined**\n\n"
            "After manual review by a support supervisor, this refund request could not be approved automatically."
        )
        if note:
            msg += f"\n\n*Reason:* {note}"

    return {
        "final_response": msg,
        "steps_taken": steps,
    }


def general_node(state: SupportState) -> dict:
    """Handle general questions."""
    llm = _get_llm()
    reply = llm.invoke(
        "You are a friendly Customer Support Assistant. Answer concisely.\n\n"
        f"Customer query: {state['question']}"
    )
    text = _extract_text(reply.content)
    steps = list(state.get("steps_taken") or [])
    steps.append("general_node")
    return {
        "draft_response": text,
        "final_response": text,
        "steps_taken": steps,
    }


# --------------------------------------------------------------------------
# 3. Conditional Routing
# --------------------------------------------------------------------------
def route_by_category(state: SupportState) -> Literal["order", "refund_prepare", "general"]:
    cat = state.get("category", "general")
    if cat == "order":
        return "order"
    if cat == "refund":
        return "refund_prepare"
    return "general"


# --------------------------------------------------------------------------
# 4. Graph Compilation with Checkpointer & Interrupt
# --------------------------------------------------------------------------
def build_module4_graph():
    builder = StateGraph(SupportState)

    builder.add_node("classify", classify_node)
    builder.add_node("order", order_node)
    builder.add_node("refund_prepare", refund_prepare_node)
    builder.add_node("refund_execute", refund_execute_node)
    builder.add_node("general", general_node)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        route_by_category,
        {
            "order": "order",
            "refund_prepare": "refund_prepare",
            "general": "general",
        },
    )
    builder.add_edge("order", END)
    builder.add_edge("general", END)
    builder.add_edge("refund_prepare", "refund_execute")
    builder.add_edge("refund_execute", END)

    checkpointer = MemorySaver()
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["refund_execute"],
    )


# --------------------------------------------------------------------------
# 5. Execution & HITL Resume Helpers
# --------------------------------------------------------------------------
def _format_output(graph, config: dict) -> dict:
    snapshot = graph.get_state(config)
    values = snapshot.values or {}
    next_nodes = list(snapshot.next or ())
    interrupted = "refund_execute" in next_nodes

    draft_text = _extract_text(values.get("draft_response", ""))
    final_text = _extract_text(values.get("final_response", ""))
    response_text = draft_text if interrupted else final_text

    return {
        "thread_id": config["configurable"]["thread_id"],
        "category": values.get("category", "general"),
        "response": response_text,
        "draft_response": draft_text,
        "requires_approval": interrupted,
        "approved": values.get("approved"),
        "steps_taken": values.get("steps_taken", []),
        "turn_count": values.get("turn_count", 0),
        "next_nodes": next_nodes,
    }


def run_workflow(question: str, thread_id: str, graph) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    initial_input: SupportState = {
        "question": question,
        "category": "",
        "requires_approval": False,
        "approved": None,
        "reviewer_note": "",
        "draft_response": "",
        "final_response": "",
        "steps_taken": [],
        "turn_count": 0,
    }
    existing = graph.get_state(config)
    if existing and existing.values:
        initial_input["turn_count"] = existing.values.get("turn_count", 0)

    graph.invoke(initial_input, config=config)
    return _format_output(graph, config)


def resume_workflow(thread_id: str, approved: bool, reviewer_note: str, graph) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    if not snapshot or not snapshot.next:
        raise ValueError("No paused workflow waiting for approval on this thread_id.")

    graph.update_state(
        config,
        {
            "approved": approved,
            "reviewer_note": reviewer_note,
        },
    )
    graph.invoke(None, config=config)
    return _format_output(graph, config)


def get_workflow_state(thread_id: str, graph) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    if not snapshot or not snapshot.values:
        return {
            "thread_id": thread_id,
            "category": "",
            "response": "",
            "draft_response": "",
            "requires_approval": False,
            "approved": None,
            "steps_taken": [],
            "turn_count": 0,
            "next_nodes": [],
        }
    return _format_output(graph, config)