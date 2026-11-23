import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, File, HTTPException, UploadFile
from langgraph.checkpoint.sqlite import SqliteSaver
from Module1.service import ask_chat, create_chat_prompt, create_structured_llm
from Module2.service import ask_rag, build_rag_engine, save_uploaded_pdf
from Module3.service import ask_agent, build_module3_agent
from Module4.history_db import DEFAULT_SQLITE_DB_PATH, get_thread_messages, list_threads
from Module4.service import build_module4_graph, get_workflow_state, resume_workflow, run_workflow
from Module5.service import build_module5_graph, run_multi_agent
from backend.schemas import (
    ChatRequest,
    ChatResponse,
    MultiAgentRequest,
    MultiAgentResponse,
    RAGRequest,
    RAGResponse,
    ToolRequest,
    ToolResponse,
    WorkflowHistoryResponse,
    WorkflowRequest,
    WorkflowResponse,
    WorkflowResumeRequest,
    WorkflowThreadListResponse,
)


def _is_quota_error(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def _get_m4_graph(app: FastAPI):
    if getattr(app.state, "module4_graph", None) is None:
        app.state.module4_graph = build_module4_graph(db_path=DEFAULT_SQLITE_DB_PATH)
    return app.state.module4_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.chat_structured_llm = create_structured_llm()
    app.state.chat_prompt = create_chat_prompt()
    app.state.rag_engine = None
    app.state.module3_agent = None

    DEFAULT_SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(DEFAULT_SQLITE_DB_PATH)) as checkpointer:
        app.state.module4_graph = build_module4_graph(
            db_path=DEFAULT_SQLITE_DB_PATH,
            checkpointer=checkpointer,
        )
        app.state.module5_graph = build_module5_graph()
        yield


app = FastAPI(title="LLM Modules Shared Backend", lifespan=lifespan)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    try:
        history_turns = [{"role": t.role, "content": t.content} for t in req.history]
        result = ask_chat(
            question=req.question,
            history_turns=history_turns,
            structured_llm=app.state.chat_structured_llm,
            prompt=app.state.chat_prompt,
        )
        return ChatResponse(**result)
    except Exception as exc:
        if _is_quota_error(exc):
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/rag/upload")
async def rag_upload_endpoint(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a valid .pdf file.")
    try:
        pdf_bytes = await file.read()
        saved_path = save_uploaded_pdf(pdf_bytes, filename=file.filename)
        app.state.rag_engine = build_rag_engine(pdf_path=saved_path)
        return {"status": "indexed", "filename": file.filename}
    except Exception as exc:
        if _is_quota_error(exc):
            raise HTTPException(
                status_code=429,
                detail="Gemini daily embedding quota exceeded (1,000/day on Free Tier). Switch API key/project or retry after quota reset.",
            ) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/rag", response_model=RAGResponse)
def rag_endpoint(req: RAGRequest):
    try:
        rag_engine = getattr(app.state, "rag_engine", None)
        if rag_engine is None:
            raise HTTPException(status_code=400, detail="No PDF indexed yet. Upload a PDF first.")
        result = ask_rag(rag_engine, req.question)
        return RAGResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        if _is_quota_error(exc):
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/tools", response_model=ToolResponse)
def tools_endpoint(req: ToolRequest):
    try:
        if getattr(app.state, "module3_agent", None) is None:
            app.state.module3_agent = build_module3_agent()
        result = ask_agent(req.prompt, agent=app.state.module3_agent)
        return ToolResponse(**result)
    except Exception as exc:
        if _is_quota_error(exc):
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/workflow", response_model=WorkflowResponse)
def workflow_endpoint(req: WorkflowRequest):
    try:
        result = run_workflow(
            question=req.question,
            thread_id=req.thread_id,
            graph=_get_m4_graph(app),
            db_path=DEFAULT_SQLITE_DB_PATH,
        )
        return WorkflowResponse(**result)
    except Exception as exc:
        if _is_quota_error(exc):
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/workflow/resume", response_model=WorkflowResponse)
def workflow_resume_endpoint(req: WorkflowResumeRequest):
    try:
        result = resume_workflow(
            thread_id=req.thread_id,
            approved=req.approved,
            reviewer_note=req.reviewer_note,
            graph=_get_m4_graph(app),
            db_path=DEFAULT_SQLITE_DB_PATH,
        )
        return WorkflowResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        if _is_quota_error(exc):
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/workflow/state/{thread_id}", response_model=WorkflowResponse)
def workflow_state_endpoint(thread_id: str):
    try:
        result = get_workflow_state(thread_id=thread_id, graph=_get_m4_graph(app))
        return WorkflowResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/workflow/threads", response_model=WorkflowThreadListResponse)
def workflow_threads_endpoint():
    try:
        return WorkflowThreadListResponse(threads=list_threads(db_path=DEFAULT_SQLITE_DB_PATH))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/workflow/history/{thread_id}", response_model=WorkflowHistoryResponse)
def workflow_history_endpoint(thread_id: str):
    try:
        graph = _get_m4_graph(app)
        messages = get_thread_messages(thread_id=thread_id, db_path=DEFAULT_SQLITE_DB_PATH)
        state = get_workflow_state(thread_id=thread_id, graph=graph)
        return WorkflowHistoryResponse(
            thread_id=thread_id,
            messages=messages,
            state=WorkflowResponse(**state),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/multi-agent", response_model=MultiAgentResponse)
def multi_agent_endpoint(req: MultiAgentRequest):
    try:
        if getattr(app.state, "module5_graph", None) is None:
            app.state.module5_graph = build_module5_graph()
        result = run_multi_agent(request=req.request, graph=app.state.module5_graph)
        return MultiAgentResponse(**result)
    except Exception as exc:
        if _is_quota_error(exc):
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc