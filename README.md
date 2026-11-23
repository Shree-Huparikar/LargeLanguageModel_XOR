# LangChain Training – Modules 1, 2 and 3

Three practical LangChain + Google Gemini projects (`gemini-3.5-flash-lite`, `gemini-embedding-001`), served by a shared FastAPI backend and a professional Streamlit UI for each module.

## Project Structure

LargeLangugaeModel_XOR/
├── .env
├── .gitignore
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── README.md
├── Module1/                # Chatbot with structured output
│   ├── __init__.py
│   ├── main.py
│   └── service.py
├── Module2/                # PDF Q&A using RAG
│   ├── __init__.py
│   ├── main.py
│   ├── service.py
│   └── sample_document.pdf
├── Module3/                # Agent with tool calling
│   ├── __init__.py
│   ├── main.py
│   └── service.py
├── Module4/                # Stateful workflow & HITL using LangGraph
│   ├── __init__.py
│   ├── main.py             # Standalone CLI runner
│   ├── service.py          # StateGraph, routing, SqliteSaver & HITL
│   └── history_db.py       # SQLite message & thread history store
├── Module5/                # Multi-Agent Travel Planner (Supervisor + Specialists)
│   ├── __init__.py
│   ├── main.py             # Standalone CLI runner
│   └── service.py
├── backend/                # Shared FastAPI backend
│   ├── __init__.py
│   ├── main.py             # /api/chat, /api/rag, /api/tools, /api/workflow*
│   └── schemas.py          # Pydantic request/response models
└── frontends/              # Streamlit UIs
    ├── common.py           # Shared theme, sidebar, API client
    ├── module1_chat_ui.py
    ├── module2_rag_ui.py
    ├── module3_tools_ui.py
    ├── module4_langgraph.py
    └── module5_multi_agent.py

## Modules

### Module 1: Chatbot with Structured Output
- `ChatGoogleGenerativeAI`, `ChatPromptTemplate`, `MessagesPlaceholder`
- Pydantic structured output (`ChatAnswer`: `answer`, `topic`, `example`)
- Conversation history with `InMemoryChatMessageHistory`

CLI: `python Module1/main.py`

### Module 2: PDF Question Answering using RAG
- `PyPDFLoader`, `CharacterTextSplitter` (`chunk_size=300`, `chunk_overlap=0`)
- `GoogleGenerativeAIEmbeddings` (`gemini-embedding-001`), Chroma retriever (`k=4`)
- Structured output (`RAGAnswer`: `answer`, `source_snippet`)

Requires `Module2/sample_document.pdf`. CLI: `python Module2/main.py`

### Module 3: AI Agent with Tool Calling
- LangChain agent (`create_agent`) with custom `@tool` functions
- `add_numbers`, `multiply_numbers`, `get_current_date`, `get_student_info`

CLI: `python Module3/main.py`

## Backend and Frontends

| Component | Command | Port |
|---|---|---|
| FastAPI backend (`/health`, `/api/chat`, `/api/rag`, `/api/tools`) | `uvicorn backend.main:app --reload --port 8000` | 8000 |
| Chat Assistant UI | `streamlit run frontends/module1_chat_ui.py --server.port 8501` | 8501 |
| Document Q&A UI | `streamlit run frontends/module2_rag_ui.py --server.port 8502` | 8502 |
| Tool-Calling Agent UI | `streamlit run frontends/module3_tools_ui.py --server.port 8503` | 8503 |

UI features: live backend status, cross-module navigation, chat bubbles, suggested prompts, source-snippet display, tool badges and friendly error messages.

Optional environment variables: `BACKEND_URL` (default `http://localhost:8000`), `FRONTEND_HOST` (default `http://localhost`).

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_api_key_here
```

## Technologies

Python · LangChain · Google Gemini · Pydantic · Chroma · PyPDF · python-dotenv · FastAPI · Uvicorn · Streamlit · Requests


uvicorn backend.main:app --reload --port 8000
streamlit run frontends/module1_chat_ui.py --server.port 8501
streamlit run frontends/module2_rag_ui.py --server.port 8502
streamlit run frontends/module3_tools_ui.py --server.port 8503
streamlit run frontends/module4_langgraph.py --server.port 8504
streamlit run frontends/module5_multi_agent.py --server.port 8505