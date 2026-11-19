from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Load .env from the main LLM folder
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


# 1. Create Gemini model
def create_chat_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0,
    )


# 2. Define structured output
class ChatAnswer(BaseModel):
    topic: str = Field(description="Short topic label")
    answer: str = Field(description="Direct answer to the user's message or question")
    example: str = Field(description="Concrete example for concepts, or 'N/A' for personal recall")


def create_structured_llm(llm: ChatGoogleGenerativeAI | None = None):
    target_llm = llm or create_chat_llm()
    return target_llm.with_structured_output(ChatAnswer)


# 3. Create prompt template
def create_chat_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful AI assistant. Always check the conversation history first.\n"
            "- When the user states a personal fact (e.g., their name or friend's name), "
            "confirm it directly and plainly in 'answer' (e.g., 'Got it, your name is Shreyash.') "
            "and set 'example' to 'N/A'. Do NOT define the word in the abstract.\n"
            "- When the user asks a recall question about themselves, their friend, or prior turns "
            "(e.g., 'What is my name?', 'What is my friend name?'), answer with the exact fact "
            "from history directly in 'answer' (e.g., 'Your name is Shreyash.', 'Your friend's name is Sachin.') "
            "and set 'example' to 'N/A'. If unknown, say clearly that it was not mentioned.\n"
            "- When the user asks to explain a concept or topic (e.g., 'Explain about oval'), "
            "give a clear, direct explanation in 'answer' and one concrete example in 'example'.",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])


# 4. Create conversation history
def create_chat_history() -> InMemoryChatMessageHistory:
    return InMemoryChatMessageHistory()


def rebuild_history(history_turns: list[dict[str, str]] | None = None) -> InMemoryChatMessageHistory:
    chat_history = InMemoryChatMessageHistory()
    for turn in history_turns or []:
        role = turn.get("role", "")
        content = extract_text_content(turn.get("content", ""))
        if not content:
            continue
        if role == "user":
            chat_history.add_user_message(content)
        elif role in ("assistant", "ai"):
            chat_history.add_ai_message(content)
    return chat_history


def ask_chat(
    question: str,
    history_turns: list[dict[str, str]] | None = None,
    structured_llm=None,
    prompt: ChatPromptTemplate | None = None,
) -> dict[str, str]:
    target_structured_llm = structured_llm or create_structured_llm()
    target_prompt = prompt or create_chat_prompt()
    clean_question = extract_text_content(question)
    chat_history = rebuild_history(history_turns)

    messages = target_prompt.invoke({
        "history": chat_history.messages,
        "question": clean_question,
    })

    response: ChatAnswer = target_structured_llm.invoke(messages)

    clean_topic = extract_text_content(response.topic)
    clean_answer = extract_text_content(response.answer)
    clean_example = extract_text_content(response.example)

    readable_ai_turn = (
        f"Topic: {clean_topic}\n"
        f"Answer: {clean_answer}\n"
        f"Example: {clean_example}"
    )

    chat_history.add_user_message(clean_question)
    chat_history.add_ai_message(readable_ai_turn)

    return {
        "topic": clean_topic,
        "answer": clean_answer,
        "example": clean_example,
        "ai_history_entry": readable_ai_turn,
    }