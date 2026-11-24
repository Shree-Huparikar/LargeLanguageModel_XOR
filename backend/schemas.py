from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatTurn] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    topic: str = "General"
    example: str = ""
    ai_history_entry: str = ""


class RAGRequest(BaseModel):
    question: str


class RAGResponse(BaseModel):
    answer: str
    source_snippet: str = ""
    citations: str = ""
    sources: list[str] = Field(default_factory=list)
    ai_history_entry: str = ""


class ToolRequest(BaseModel):
    prompt: str


class ToolResponse(BaseModel):
    answer: str = ""
    response: str = ""
    result: str = ""
    tools_used: list[str] = Field(default_factory=list)
    ai_history_entry: str = ""


# --- Module 4: LangGraph Workflow & HITL ---
class WorkflowRequest(BaseModel):
    question: str
    thread_id: str = "default-thread"


class WorkflowResumeRequest(BaseModel):
    thread_id: str
    approved: bool
    reviewer_note: str = ""


class WorkflowResponse(BaseModel):
    thread_id: str
    category: str = ""
    response: str = ""
    draft_response: str = ""
    requires_approval: bool = False
    approved: bool | None = None
    steps_taken: list[str] = Field(default_factory=list)
    turn_count: int = 0
    next_nodes: list[str] = Field(default_factory=list)


class WorkflowMessageItem(BaseModel):
    role: str
    content: str
    response: str = ""
    category: str = ""
    draft_response: str = ""
    requires_approval: bool = False
    steps_taken: list[str] = Field(default_factory=list)
    turn_count: int = 0
    created_at: str = ""


class WorkflowHistoryResponse(BaseModel):
    thread_id: str
    messages: list[WorkflowMessageItem] = Field(default_factory=list)
    state: WorkflowResponse


class WorkflowThreadSummary(BaseModel):
    thread_id: str
    thread_num: int = 1
    title: str = "Thread 1"
    message_count: int = 0
    last_activity: str = ""
    last_question: str = ""
    last_category: str = ""


class WorkflowThreadListResponse(BaseModel):
    threads: list[WorkflowThreadSummary] = Field(default_factory=list)


# --- Module 5: Multi-Agent Systems ---
class MultiAgentRequest(BaseModel):
    request: str


class MultiAgentResponse(BaseModel):
    result: str = ""
    travel: str = ""
    budget: str = ""
    booking: str = ""
    activities: str = ""
    agents_used: list[str] = Field(default_factory=list)
    handoff_log: list[str] = Field(default_factory=list)