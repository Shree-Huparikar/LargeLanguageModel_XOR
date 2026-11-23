"""Module 4 - LangGraph Stateful Support Workflow with SQLite Checkpointing & HITL."""
import os
from pathlib import Path
from typing import Any, Literal, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from Module4.history_db import DEFAULT_SQLITE_DB_PATH, init_history_db, save_message


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
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        temperature=0.2,
    )


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
            else:
                parts.append(getattr(block, "text", "") or "")
        return "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    return "" if content is None else str(content).strip()


def _append_step(state: SupportState, step: str) -> list[str]:
    return [*(state.get("steps_taken") or []), step]


def classify_node(state: SupportState) -> dict:
    decision: RouteDecision = _get_llm().with_structured_output(RouteDecision).invoke(
        "You are a customer support triage router.\n"
        "Classify the user's message into one of: 'order', 'refund', or 'general'.\n"
        "Set requires_approval=True whenever category is 'refund'.\n\n"
        f"User message: {state['question']}"
    )
    return {
        "category": decision.category,
        "requires_approval": decision.requires_approval,
        "approved": None,
        "reviewer_note": "",
        "steps_taken": _append_step(state, f"classify -> {decision.category}"),
        "turn_count": (state.get("turn_count") or 0) + 1,
    }


def order_node(state: SupportState) -> dict:
    reply = _get_llm().invoke(
        "You are an Order Support Specialist. Give a helpful, concise response "
        "explaining how to check order status/tracking and next steps.\n\n"
        f"Customer query: {state['question']}"
    )
    text = _extract_text(reply.content)
    return {"draft_response": text, "final_response": text, "steps_taken": _append_step(state, "order_node")}


def refund_prepare_node(state: SupportState) -> dict:
    draft = _get_llm().invoke(
        "You are a Refund Specialist drafting a proposed refund resolution for manager review. "
        "Summarize the refund request and proposed action in 2-3 sentences.\n\n"
        f"Customer query: {state['question']}"
    )
    text = _extract_text(draft.content)
    return {
        "draft_response": text,
        "steps_taken": _append_step(state, "refund_prepare (awaiting human approval)"),
    }


def refund_execute_node(state: SupportState) -> dict:
    note = (state.get("reviewer_note") or "").strip()
    draft_text = _extract_text(state.get("draft_response", ""))
    if state.get("approved"):
        step = "refund_execute (APPROVED)"
        msg = (
            f"**Refund Approved**\n\n{draft_text}\n\n"
            "Your refund has been authorized by a support supervisor and will reflect within 3-5 business days."
        )
        if note:
            msg += f"\n\n*Supervisor note:* {note}"
    else:
        step = "refund_execute (REJECTED)"
        msg = (
            "**Refund Request Declined**\n\n"
            "After manual review by a support supervisor, this refund request could not be approved automatically."
        )
        if note:
            msg += f"\n\n*Reason:* {note}"
    return {"final_response": msg, "steps_taken": _append_step(state, step)}


def general_node(state: SupportState) -> dict:
    reply = _get_llm().invoke(
        "You are a friendly Customer Support Assistant. Answer concisely.\n\n"
        f"Customer query: {state['question']}"
    )
    text = _extract_text(reply.content)
    return {"draft_response": text, "final_response": text, "steps_taken": _append_step(state, "general_node")}


def route_by_category(state: SupportState) -> Literal["order", "refund_prepare", "general"]:
    return {"order": "order", "refund": "refund_prepare"}.get(state.get("category", "general"), "general")


def build_module4_graph(
    db_path: str | Path = DEFAULT_SQLITE_DB_PATH,
    checkpointer: SqliteSaver | None = None,
):
    init_history_db(db_path)
    builder = StateGraph(SupportState)
    for name, fn in [
        ("classify", classify_node),
        ("order", order_node),
        ("refund_prepare", refund_prepare_node),
        ("refund_execute", refund_execute_node),
        ("general", general_node),
    ]:
        builder.add_node(name, fn)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        route_by_category,
        {"order": "order", "refund_prepare": "refund_prepare", "general": "general"},
    )
    builder.add_edge("order", END)
    builder.add_edge("general", END)
    builder.add_edge("refund_prepare", "refund_execute")
    builder.add_edge("refund_execute", END)

    if checkpointer is not None:
        return builder.compile(checkpointer=checkpointer, interrupt_before=["refund_execute"])

    resolved = Path(db_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    saver_cm = SqliteSaver.from_conn_string(str(resolved))
    compiled = builder.compile(checkpointer=saver_cm.__enter__(), interrupt_before=["refund_execute"])
    compiled._sqlite_cm = saver_cm  # type: ignore[attr-defined]
    return compiled


def _format_output(graph, config: dict) -> dict:
    snapshot = graph.get_state(config)
    values = snapshot.values or {}
    next_nodes = list(snapshot.next or ())
    interrupted = "refund_execute" in next_nodes
    draft_text = _extract_text(values.get("draft_response", ""))
    final_text = _extract_text(values.get("final_response", ""))
    return {
        "thread_id": config["configurable"]["thread_id"],
        "category": values.get("category", "general"),
        "response": draft_text if interrupted else final_text,
        "draft_response": draft_text,
        "requires_approval": interrupted,
        "approved": values.get("approved"),
        "steps_taken": values.get("steps_taken", []),
        "turn_count": values.get("turn_count", 0),
        "next_nodes": next_nodes,
    }


def run_workflow(
    question: str,
    thread_id: str,
    graph,
    db_path: str | Path = DEFAULT_SQLITE_DB_PATH,
) -> dict:
    save_message(thread_id=thread_id, role="user", content=question, db_path=db_path)
    config = {"configurable": {"thread_id": thread_id}}
    existing = graph.get_state(config)
    initial_input: SupportState = {
        "question": question,
        "category": "",
        "requires_approval": False,
        "approved": None,
        "reviewer_note": "",
        "draft_response": "",
        "final_response": "",
        "steps_taken": [],
        "turn_count": existing.values.get("turn_count", 0) if existing and existing.values else 0,
    }
    graph.invoke(initial_input, config=config)
    out = _format_output(graph, config)
    save_message(
        thread_id=thread_id,
        role="assistant",
        content=out["response"] or out["draft_response"],
        category=out["category"],
        draft_response=out["draft_response"],
        requires_approval=out["requires_approval"],
        steps_taken=out["steps_taken"],
        turn_count=out["turn_count"],
        db_path=db_path,
    )
    return out


def resume_workflow(
    thread_id: str,
    approved: bool,
    reviewer_note: str,
    graph,
    db_path: str | Path = DEFAULT_SQLITE_DB_PATH,
) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    if not snapshot or not snapshot.next:
        raise ValueError("No paused workflow waiting for approval on this thread_id.")
    graph.update_state(config, {"approved": approved, "reviewer_note": reviewer_note})
    graph.invoke(None, config=config)
    out = _format_output(graph, config)
    save_message(
        thread_id=thread_id,
        role="assistant",
        content=out["response"],
        category=out["category"],
        draft_response=out["draft_response"],
        requires_approval=False,
        steps_taken=out["steps_taken"],
        turn_count=out["turn_count"],
        db_path=db_path,
    )
    return out


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