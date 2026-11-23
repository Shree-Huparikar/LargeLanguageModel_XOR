# LangChain & LangGraph Training – Modules 1 to 5

A comprehensive collection of practical LangChain, LangGraph, and Google Gemini projects, served by a unified FastAPI backend and interactive Streamlit UIs.

## 📁 Project Structure

```
LargeLanguageModel_XOR/
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
```

## 🧩 Modules Overview

### Module 1: Chatbot with Structured Output
- **Key Concepts:** `ChatGoogleGenerativeAI`, `ChatPromptTemplate`, `MessagesPlaceholder`
- **Structured Output:** Pydantic schema (`ChatAnswer`: `answer`, `topic`, `example`)
- **Memory:** Conversation history maintained with `InMemoryChatMessageHistory`
- **CLI:** `python Module1/main.py`

### Module 2: PDF Question Answering using RAG
- **Key Concepts:** `PyPDFLoader`, `CharacterTextSplitter` (`chunk_size=300`, `chunk_overlap=0`)
- **Vectorstore:** `GoogleGenerativeAIEmbeddings` (`gemini-embedding-001`), Chroma retriever (`k=4`)
- **Structured Output:** Pydantic schema (`RAGAnswer`: `answer`, `source_snippet`)
- **CLI:** `python Module2/main.py` *(Requires `Module2/sample_document.pdf`)*

### Module 3: AI Agent with Tool Calling
- **Key Concepts:** LangChain Agent (`create_agent`) with `@tool` integrations
- **Tools:** `add_numbers`, `multiply_numbers`, `get_current_date`, `get_student_info`
- **CLI:** `python Module3/main.py`

### Module 4: Stateful Support Workflow & HITL (LangGraph)
- **Key Concepts:** `StateGraph`, conditional routing (`order`, `refund`, `general`)
- **Persistence:** `SqliteSaver` checkpointer and `workflow_messages` history DB
- **HITL:** Human-in-the-Loop review & approval gate before issuing refunds
- **CLI:** `python Module4/main.py`

### Module 5: Multi-Agent Travel Planner
- **Key Concepts:** Multi-agent collaboration with supervisor orchestration and specialist handoffs
- **Agents:** Travel Specialist, Budget Specialist, Booking Specialist, Activities Specialist
- **CLI:** `python Module5/main.py`

## 🚀 Running the Application

### Backend & Frontends Ports

| Component | Command | Port |
|---|---|---|
| **FastAPI Backend** | `uvicorn backend.main:app --reload --port 8000` | `8000` |
| **Module 1 UI (Chat)** | `streamlit run frontends/module1_chat_ui.py --server.port 8501` | `8501` |
| **Module 2 UI (RAG)** | `streamlit run frontends/module2_rag_ui.py --server.port 8502` | `8502` |
| **Module 3 UI (Tools)** | `streamlit run frontends/module3_tools_ui.py --server.port 8503` | `8503` |
| **Module 4 UI (LangGraph)** | `streamlit run frontends/module4_langgraph.py --server.port 8504` | `8504` |
| **Module 5 UI (Multi-Agent)** | `streamlit run frontends/module5_multi_agent.py --server.port 8505` | `8505` |

## ⚙️ Setup & Installation

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv

# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# Upgrade pip and install requirements
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-3.5-flash
BACKEND_URL=http://localhost:8000
```

## 🛠️ Technology Stack

**Languages & Frameworks:** Python · LangChain · LangGraph · FastAPI · Streamlit  
**Models & Embeddings:** Google Gemini · `gemini-embedding-001`  
**Databases & Storage:** SQLite (`SqliteSaver`) · Chroma Vector DB  
**Utilities:** Pydantic · PyPDF · python-dotenv · Uvicorn · Requests