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