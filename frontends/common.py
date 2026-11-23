"""Shared UI helpers for all Streamlit frontends (theme, sidebar, API client)."""
import asyncio
import os
import sys
import time
import uuid
from html import escape

import requests
import streamlit as st

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
FRONTEND_HOST = os.getenv("FRONTEND_HOST", "http://localhost")

# Request field names expected by backend/schemas.py.
# If you get a 422 error, change these to match your Pydantic request models.
REQUEST_FIELDS = {
    "chat": "question",
    "rag": "question",
    "tools": "message",
    "workflow": "question",
    "multi_agent": "request",
}

MODULES = {
    "chat": {"label": "Chat Assistant", "icon": "💬", "port": 8501, "endpoint": "/api/chat"},
    "rag": {"label": "Document Q&A (RAG)", "icon": "📄", "port": 8502, "endpoint": "/api/rag"},
    "tools": {"label": "Tool-Calling Agent", "icon": "🛠️", "port": 8503, "endpoint": "/api/tools"},
    "workflow": {"label": "LangGraph Workflow", "icon": "🔀", "port": 8504, "endpoint": "/api/workflow"},
    "multi_agent": {"label": "Multi-Agent Planner", "icon": "🤝", "port": 8505, "endpoint": "/api/multi-agent"},
}

CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 960px;}

.app-hero {
    padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1.2rem;
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    color: #fff;
}
.app-hero h1 {margin: 0; font-size: 1.7rem; font-weight: 700; color: #fff; padding: 0;}
.app-hero p {margin: .35rem 0 0 0; opacity: .9; font-size: .98rem; color: #fff;}

.chip {
    display: inline-block; padding: 2px 10px; margin: 0 6px 6px 0;
    border-radius: 999px; font-size: .78rem; font-weight: 600;
    background: rgba(79,70,229,.12); color: #6366F1;
    border: 1px solid rgba(79,70,229,.25);
}
.chip.green {background: rgba(16,185,129,.12); color: #10B981; border-color: rgba(16,185,129,.3);}
.chip.amber {background: rgba(245,158,11,.12); color: #F59E0B; border-color: rgba(245,158,11,.3);}

.status {display:inline-flex; align-items:center; gap:8px; font-size:.85rem; font-weight:600;}
.dot {width:9px; height:9px; border-radius:50%; display:inline-block;}
.dot.on {background:#10B981; box-shadow:0 0 0 3px rgba(16,185,129,.2);}
.dot.off {background:#EF4444; box-shadow:0 0 0 3px rgba(239,68,68,.2);}

.quote {
    border-left: 4px solid #6366F1; padding: .7rem 1rem; border-radius: 0 10px 10px 0;
    background: rgba(99,102,241,.07); font-size: .92rem; line-height: 1.55;
    white-space: pre-wrap;
}
.muted {opacity: .65; font-size: .8rem;}
.empty-state {text-align:center; padding: 2.5rem 1rem; opacity:.75;}
.empty-state .emoji {font-size: 2.4rem;}

div[data-testid="stSidebar"] .stButton button {
    width: 100%; text-align: left; border-radius: 10px;
}
</style>
"""


# --------------------------------------------------------------------------
# Page + layout
# --------------------------------------------------------------------------
def setup_page(module_key: str) -> None:
    meta = MODULES[module_key]
    st.set_page_config(
        page_title=f"{meta['label']} · LangChain Training",
        page_icon=meta["icon"],
        layout="centered",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("backend_url", DEFAULT_BACKEND_URL)
    st.session_state.setdefault("session_id", str(uuid.uuid4()))


def hero(title: str, subtitle: str, icon: str = "") -> None:
    st.markdown(
        f'<div class="app-hero"><h1>{icon} {escape(title)}</h1><p>{escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def empty_state(emoji: str, text: str) -> None:
    st.markdown(
        f'<div class="empty-state"><div class="emoji">{emoji}</div><p>{escape(text)}</p></div>',
        unsafe_allow_html=True,
    )


def chips(items, style: str = "") -> None:
    html = "".join(f'<span class="chip {style}">{escape(str(i))}</span>' for i in items if i)
    if html:
        st.markdown(html, unsafe_allow_html=True)


def quote(text: str) -> None:
    st.markdown(f'<div class="quote">{escape(text)}</div>', unsafe_allow_html=True)


def sidebar(module_key: str) -> None:
    """Brand, backend status, module navigation, settings."""
    with st.sidebar:
        st.markdown("### 🧠 LangChain Training")
        st.caption("Powered by Google Gemini · FastAPI · Streamlit")

        online = check_health(st.session_state["backend_url"])
        dot = "on" if online else "off"
        label = "Backend online" if online else "Backend offline"
        st.markdown(
            f'<span class="status"><span class="dot {dot}"></span>{label}</span>',
            unsafe_allow_html=True,
        )
        if not online:
            st.caption("Start it with: `uvicorn backend.main:app --reload --port 8000`")

        st.divider()
        st.markdown("**Modules**")
        for key, m in MODULES.items():
            if key == module_key:
                st.markdown(f"**{m['icon']} {m['label']}** · _current_")
            else:
                st.link_button(f"{m['icon']} {m['label']}", f"{FRONTEND_HOST}:{m['port']}", use_container_width=True)

        st.divider()
        with st.expander("⚙️ Settings"):
            st.session_state["backend_url"] = st.text_input(
                "Backend URL", st.session_state["backend_url"]
            ).rstrip("/")
            if st.button("Re-check connection"):
                check_health.clear()
                st.rerun()


# --------------------------------------------------------------------------
# API client
# --------------------------------------------------------------------------
@st.cache_data(ttl=10, show_spinner=False)
def check_health(base_url: str) -> bool:
    try:
        return requests.get(f"{base_url}/health", timeout=3).ok
    except requests.RequestException:
        return False


def unwrap(data):
    """Tolerate responses wrapped as {"data": {...}} / {"result": {...}}."""
    if isinstance(data, dict):
        for key in ("data", "result", "response"):
            if isinstance(data.get(key), dict):
                return data[key]
    return data


def pick(data, *keys, default=""):
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        for k in keys:
            if data.get(k) not in (None, ""):
                return data[k]
    return default


def _missing_field(resp):
    """If a 422 says a body field is missing, return that field's name."""
    try:
        for err in resp.json().get("detail", []):
            if err.get("type") == "missing" and err.get("loc", [None])[0] == "body":
                return err["loc"][-1]
    except Exception:  # noqa: BLE001
        pass
    return None


def api_post(path: str, payload: dict):
    """Returns (data, seconds, error_message).

    The first key in `payload` holds the user's text. If the backend replies
    422 "field required" for a different name, the request is retried once with
    that name, and the correction is remembered for the rest of the session.
    """
    url = f"{st.session_state['backend_url']}{path}"
    overrides = st.session_state.setdefault("field_overrides", {})
    primary = next(iter(payload))
    text = payload[primary]
    rest = {k: v for k, v in payload.items() if k != primary}
    body = {overrides.get(path, primary): text, **rest}

    start = time.perf_counter()
    try:
        resp = requests.post(url, json=body, timeout=120)
        if resp.status_code == 422:
            wanted = _missing_field(resp)
            if wanted and wanted not in body:
                overrides[path] = wanted
                body = {wanted: text, **rest}
                resp = requests.post(url, json=body, timeout=120)
        resp.raise_for_status()
        return unwrap(resp.json()), time.perf_counter() - start, None
    except requests.exceptions.ConnectionError:
        return None, 0, "Cannot reach the backend. Make sure FastAPI is running on the configured URL."
    except requests.exceptions.Timeout:
        return None, 0, "The request timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", e.response.text)
        except Exception:  # noqa: BLE001
            detail = e.response.text
        return None, 0, f"Backend returned {e.response.status_code}: {detail}"
    except Exception as e:  # noqa: BLE001
        return None, 0, f"Unexpected error: {e}"


def api_upload(path: str, filename: str, content: bytes, mime: str = "application/pdf"):
    """Multipart file upload. Returns (data, seconds, error_message)."""
    url = f"{st.session_state['backend_url']}{path}"
    start = time.perf_counter()
    try:
        resp = requests.post(url, files={"file": (filename, content, mime)}, timeout=300)
        resp.raise_for_status()
        return unwrap(resp.json()), time.perf_counter() - start, None
    except requests.exceptions.ConnectionError:
        return None, 0, "Cannot reach the backend. Make sure FastAPI is running on the configured URL."
    except requests.exceptions.Timeout:
        return None, 0, "Indexing timed out. Try a smaller PDF."
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", e.response.text)
        except Exception:  # noqa: BLE001
            detail = e.response.text
        return None, 0, f"Backend returned {e.response.status_code}: {detail}"
    except Exception as e:  # noqa: BLE001
        return None, 0, f"Unexpected error: {e}"
