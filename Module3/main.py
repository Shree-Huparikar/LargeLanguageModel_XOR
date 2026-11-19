# import os
# from pathlib import Path
# from dotenv import load_dotenv
# from datetime import datetime
 
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.tools import tool
# from langchain.agents import create_agent
 
# # Load .env from the main LLM folder
# ROOT_DIR = Path(__file__).resolve().parent.parent
# load_dotenv(ROOT_DIR / ".env")
 
# # 1. Create Gemini model
# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.5-flash-lite",
#     temperature=0
# )
 
# # 2. Addition tool
# @tool
# def add_numbers(a: int, b: int) -> int:
#     """Add two numbers."""
#     return a + b
 
# # 3. Multiplication tool
# @tool
# def multiply_numbers(a: int, b: int) -> int:
#     """Multiply two numbers."""
#     return a * b
 
# # 4. Current date tool
# @tool
# def get_current_date() -> str:
#     """Get today's current date."""
#     return datetime.now().strftime("%d-%m-%Y")
 
# # 5. Student information tool
# @tool
# def get_student_info(name: str) -> str:
#     """Get student information by name."""
 
#     students = {
#         "shreyash": "Computer Engineering student with 8.95 CGPA",
#         "rahul": "Computer Engineering student with 8.5 CGPA"
#     }
 
#     name = name.lower()
 
#     if name in students:
#         return students[name]
 
#     return "Student not found."
 
# # 6. Register tools
# tools = [
#     add_numbers,
#     multiply_numbers,
#     get_current_date,
#     get_student_info
# ]
 
# # 7. Create AI agent
# agent = create_agent(
#     model=llm,
#     tools=tools,
#     system_prompt=(
#         "You are a helpful AI personal assistant. "
#         "Use available tools whenever required. "
#         "Give simple and clear answers."
#     )
# )
 
# # 8. Run agent
# while True:
#     user_input = input(
#         "\nAsk something (type exit to stop): "
#     )
 
#     if user_input.lower() == "exit":
#         print("Agent stopped.")
#         break
 
#     response = agent.invoke({
#         "messages": [
#             {"role": "user", "content": user_input}
#         ]
#     })
 
#     print(
#         "\nAI Agent:",
#         response["messages"][-1].content
#     )
