import re
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def extract_text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text") is not None:
                parts.append(str(block["text"]))
        return "".join(parts).strip()
    if isinstance(content, dict):
        return str(content.get("text", "")).strip()
    return str(content).strip()


def save_uploaded_pdf(file_bytes: bytes, filename: str = "uploaded.pdf") -> Path:
    upload_dir = Path(tempfile.gettempdir()) / "llm_rag_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name or "uploaded.pdf"
    target_path = upload_dir / safe_name
    target_path.write_bytes(file_bytes)
    return target_path


class RAGAnswer(BaseModel):
    answer: str = Field(
        description="Detailed answer formatted strictly as a numbered list (1., 2., 3.)"
    )
    source_snippet: str = Field(
        description="Exact verbatim text excerpt from the PDF context supporting the answer"
    )


def clean_and_format_points(text: str) -> str:
    text = extract_text_content(text)
    formatted = re.sub(r"(?<!^)\s+(\d+\.\s+)", r"\n\n\1", text)
    return formatted


def create_rag_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0,
    )


def create_rag_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
    )


def create_rag_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert document assistant. Answer thoroughly and in detail "
            "using only the provided context. Format your 'answer' strictly as a numbered list (1., 2., 3., etc.). "
            "For 'source_snippet', provide the exact unedited text excerpt from the document context. "
            "If the answer is not present, say you cannot find it in the document.",
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}",
        ),
    ])


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg


def _add_chunks_with_retry(
    vector_store: Chroma,
    batch: list,
    max_retries: int = 4,
    base_delay: float = 12.0,
) -> None:
    delay = base_delay
    for attempt in range(max_retries):
        try:
            vector_store.add_documents(documents=batch)
            return
        except Exception as exc:
            if _is_quota_error(exc) and attempt < max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 1.5, 65.0)
            else:
                raise


def build_rag_engine(
    pdf_path: Path | str | None = None,
    batch_size: int = 25,
    inter_batch_sleep: float = 0.5,
):
    llm = create_rag_llm()

    target_pdf = Path(pdf_path) if pdf_path else (ROOT_DIR / "Module2" / "sample_document.pdf")
    loader = PyPDFLoader(str(target_pdf))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No text chunks could be extracted from the PDF.")

    embeddings = create_rag_embeddings()
    vector_store = Chroma(embedding_function=embeddings)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        _add_chunks_with_retry(vector_store, batch)
        if i + batch_size < len(chunks) and inter_batch_sleep > 0:
            time.sleep(inter_batch_sleep)

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    structured_llm = llm.with_structured_output(RAGAnswer)
    prompt = create_rag_prompt()

    return retriever, prompt, structured_llm


def ask_rag(rag_engine, question: str) -> dict[str, str]:
    retriever, prompt, structured_llm = rag_engine

    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(extract_text_content(doc.page_content) for doc in retrieved_docs)

    messages = prompt.invoke({
        "context": context,
        "question": extract_text_content(question),
    })
    response: RAGAnswer = structured_llm.invoke(messages)

    return {
        "answer": clean_and_format_points(response.answer),
        "source_snippet": extract_text_content(response.source_snippet),
    }