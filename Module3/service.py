import os
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

STUDENTS = {
    "101": {"name": "Shreyash", "course": "B.Tech CS", "gpa": 8.9},
    "102": {"name": "Rishi", "course": "Data Science", "gpa": 9.2},
    "103": {"name": "Achal", "course": "AI & ML", "gpa": 8.6},
}

HEADERS = {"User-Agent": "LLMTrainingAgent/1.0 (local-learning-app)"}


def extract_text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in content
        ]
        return "".join(parts).strip()
    if isinstance(content, dict):
        return str(content.get("text", "")).strip()
    return str(content).strip()


# 1. Simple calculator tool
@tool
def calculator(expression: str) -> str:
    """Calculate a simple math expression like '12 + 8' or '25 * 4'."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except ZeroDivisionError:
        return "Cannot divide by zero."
    except Exception:
        return "Invalid math expression."


# 2. External API tool for current date
@tool
def get_current_date(timezone: str = "Asia/Kolkata") -> str:
    """Fetch today's date from an external API."""
    try:
        res = requests.get(
            "https://timeapi.io/api/time/current/zone",
            params={"timeZone": timezone},
            timeout=10,
        )
        res.raise_for_status()
        return res.json().get("date") or datetime.now().strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")


# 3. Custom tool for student info
@tool
def get_student_info(query: str) -> str:
    """Get student details by ID (101, 102, 103) or name."""
    q = query.strip().lower()
    for sid, info in STUDENTS.items():
        if q == sid or q in info["name"].lower():
            return f"ID: {sid}, Name: {info['name']}, Course: {info['course']}, GPA: {info['gpa']}"
    return f"Student '{query}' not found."


# 4. External API tool for people, dates of birth, and general facts
@tool
def lookup_fact(topic: str) -> str:
    """Look up information, dates of birth, or general facts about a person, place, or topic from Wikipedia."""
    clean_topic = topic.strip().replace(" ", "_")
    if not clean_topic:
        return "Please provide a topic or person to look up."
    try:
        res = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{clean_topic}",
            headers=HEADERS,
            timeout=10,
        )
        if res.status_code == 404:
            search_res = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "opensearch",
                    "search": topic.strip(),
                    "limit": 1,
                    "namespace": 0,
                    "format": "json",
                },
                headers=HEADERS,
                timeout=10,
            )
            search_res.raise_for_status()
            titles = search_res.json()[1] if len(search_res.json()) > 1 else []
            if not titles:
                return f"No information found for '{topic}'."
            best_title = titles[0].replace(" ", "_")
            res = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{best_title}",
                headers=HEADERS,
                timeout=10,
            )

        res.raise_for_status()
        data = res.json()
        extract = data.get("extract")
        title = data.get("title", topic)
        if extract:
            return f"{title}: {extract}"
        return f"No summary available for '{topic}'."
    except Exception as exc:
        return f"Could not fetch information for '{topic}': {exc}"


def build_module3_agent():
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
    return create_agent(
        model=llm,
        tools=[calculator, get_current_date, get_student_info, lookup_fact],
        system_prompt=(
            "You are a helpful assistant. Use the appropriate tool whenever possible: "
            "'calculator' for math, 'get_current_date' for today's date, "
            "'get_student_info' for student records (101-103), and 'lookup_fact' "
            "for people, birth dates, or general facts. Reply clearly in plain text."
        ),
    )


def ask_agent(
    prompt: str | None = None,
    agent: Any = None,
    history: list[dict] | None = None,
    question: str | None = None,
) -> dict[str, Any]:
    if agent is None:
        raise RuntimeError("Module 3 agent is not initialized.")

    user_input = extract_text_content(prompt if prompt is not None else question)
    if not user_input:
        raise ValueError("Prompt cannot be empty.")

    messages = []
    for turn in history or []:
        text = extract_text_content(turn.get("content", ""))
        if text:
            messages.append({"role": turn.get("role", "user"), "content": text})
    messages.append({"role": "user", "content": user_input})

    result = agent.invoke({"messages": messages})
    reply = extract_text_content(getattr(result["messages"][-1], "content", result["messages"][-1]))

    tools_used = []
    for msg in result.get("messages", []):
        for tc in getattr(msg, "tool_calls", None) or []:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name and name not in tools_used:
                tools_used.append(name)

    return {
        "answer": reply,
        "response": reply,
        "result": reply,
        "tools_used": tools_used,
        "ai_history_entry": reply,
    }