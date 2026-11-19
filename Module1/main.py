import os
from pathlib import Path
from dotenv import load_dotenv
 
from pydantic import BaseModel  #Used for data Validation and defining Structure data
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import (
    ChatPromptTemplate,    
    MessagesPlaceholder
)
from langchain_core.chat_history import InMemoryChatMessageHistory
 
# Load .env from the main LLM folder
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
 
# 1. Create Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0     #Controls the randomness of models output.
)
 
# 2. Define structured output
class ChatAnswer(BaseModel):
    answer: str
    topic: str   
    example: str
 
structured_llm = llm.with_structured_output(ChatAnswer)
 
# 3. Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Give simple answers."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])
 
# 4. Create conversation history
chat_history = InMemoryChatMessageHistory()
 
# 5. Chat loop
while True:
    question = input("\nYou: ")
 
    if question.lower() == "exit":
        print("Chat ended.")
        break
 
    messages = prompt.invoke({
        "history": chat_history.messages,
        "question": question
    })
 
    response = structured_llm.invoke(messages)
 
    print("\nAI:", response.answer)
    print("Topic:", response.topic)
    print("Example:", response.example)

# Updating Conversation History
    chat_history.add_user_message(question)
    chat_history.add_ai_message(response.model_dump_json())