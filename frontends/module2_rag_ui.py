"""Module 2 - PDF question answering with RAG. Upload your own PDF or use the sample."""
import streamlit as st

from common import (REQUEST_FIELDS, MODULES, api_post, api_upload, chips,
                    empty_state, hero, pick, quote, setup_page, sidebar)

MODULE = "rag"
setup_page(MODULE)
sidebar(MODULE)

st.session_state.setdefault("rag_history", [])
st.session_state.setdefault("rag_doc", None)        # info about the uploaded PDF
st.session_state.setdefault("rag_sig", None)        # name:size of the last handled upload
st.session_state.setdefault("rag_upload_error", None)

SAMPLE_NAME = "sample_document.pdf"

SUGGESTIONS = [
    "Summarize the main points of the document",
    "What are the key conclusions?",
    "Which topics does the document cover?",
]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("**Knowledge base**")
    uploaded = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Text-based PDF, up to 20 MB / 100 pages",
    )

    if uploaded is not None:
        sig = f"{uploaded.name}:{uploaded.size}"
        if st.session_state["rag_sig"] != sig:
            st.session_state["rag_sig"] = sig
            st.session_state["rag_upload_error"] = None
            with st.spinner("Reading and indexing your PDF…"):
                data, secs, err = api_upload("/api/rag/upload", uploaded.name, uploaded.getvalue())
            if err:
                st.session_state["rag_doc"] = None
                st.session_state["rag_upload_error"] = err
            else:
                st.session_state["rag_doc"] = data
                st.session_state["rag_history"] = []
                st.toast(f"Indexed {uploaded.name}", icon="✅")
    elif st.session_state["rag_sig"]:          # file removed with the ✕ button
        st.session_state.update(rag_sig=None, rag_doc=None, rag_upload_error=None, rag_history=[])

    doc = st.session_state["rag_doc"]
    if st.session_state["rag_upload_error"]:
        st.error(st.session_state["rag_upload_error"])

    doc_filename = None
    if isinstance(doc, dict):
        fallback_name = uploaded.name if uploaded is not None else SAMPLE_NAME
        doc_filename = pick(doc, "filename", "file_name", "name", default=fallback_name)
        pages = pick(doc, "pages", "num_pages", "page_count", default="?")
        chunks = pick(doc, "chunks", "num_chunks", "chunk_count", default=0)
        st.success(f"📄 {doc_filename}")
        st.caption(f"{pages} pages · {chunks} chunks indexed")
    else:
        st.markdown(f"📄 `{SAMPLE_NAME}` (default)")
        st.caption("Upload a PDF above to ask about your own document.")

    st.divider()
    st.markdown("**Sample questions**")
    for s in SUGGESTIONS:
        if st.button(s, key=f"sugg_{s}"):
            st.session_state["queued"] = s
    st.divider()
    if st.button("🗑️ Clear history"):
        st.session_state["rag_history"] = []
        st.rerun()

# ------------------------------------------------------------------- main
hero(
    "Document Q&A",
    "Upload a PDF and ask questions. Every answer is grounded in a source passage.",
    MODULES[MODULE]["icon"],
)
active_name = doc_filename or SAMPLE_NAME
chips([f"📄 Active document: {active_name}"], "green" if isinstance(doc, dict) else "")


def render(item: dict, expanded: bool = True) -> None:
    st.markdown(f"##### ❓ {item['question']}")
    st.markdown(item["answer"])
    if item.get("source"):
        with st.expander("📎 Source snippet", expanded=expanded):
            quote(item["source"])
    chips([f"📄 {item.get('doc', '')}", f"⏱ {item['latency']:.1f}s"], "green")


with st.form("rag_form", clear_on_submit=True, border=True):
    question = st.text_area(
        "Your question",
        placeholder="e.g. What does the document say about…?",
        height=90,
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Ask document", type="primary", use_container_width=True)

if not question.strip() and st.session_state.get("queued"):
    question, submitted = st.session_state.pop("queued"), True

if submitted and question.strip():
    q = question.strip()
    doc_id = pick(doc, "doc_id", "id", "document_id") if isinstance(doc, dict) else None
    if doc_id:
        path, payload = "/api/rag/ask", {REQUEST_FIELDS["rag"]: q, "doc_id": doc_id}
    else:
        path, payload = MODULES[MODULE]["endpoint"], {REQUEST_FIELDS["rag"]: q}

    with st.spinner("Searching the document…"):
        data, secs, err = api_post(path, payload)

    if err and isinstance(doc, dict) and "404" in err:
        # Backend restarted and lost the in-memory index: re-index automatically, then re-ask.
        st.session_state.update(rag_sig=None, rag_doc=None, queued=q)
        st.rerun()
    elif err:
        st.error(err)
    else:
        st.session_state["rag_history"].insert(0, {
            "question": q,
            "answer": pick(data, "answer", "response", "output", default="_No answer returned._"),
            "source": pick(data, "source_snippet", "source", "snippet"),
            "latency": secs,
            "doc": active_name,
        })

history = st.session_state["rag_history"]
if not history:
    empty_state("📄", "Ask a question above to get an answer with its supporting source.")
else:
    render(history[0])
    if len(history) > 1:
        st.divider()
        st.markdown("**Previous questions**")
        for item in history[1:]:
            with st.expander(item["question"]):
                render({**item, "question": "Answer"}, expanded=False)