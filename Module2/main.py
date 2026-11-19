import os
import warnings   #To hide the annoying terminal warnings
import re
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Suppress deprecation and runtime warnings for a clean terminal output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
 
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
 
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
 
# Load .env from the main LLM folder
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
 
#1. Create Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0
)
 
#2. Load PDF
pdf_path = ROOT_DIR / "Module2" / "sample_document.pdf"
 
loader = PyPDFLoader(str(pdf_path))
documents = loader.load()
 
#3. Split PDF into chunks
splitter = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0
)
 
chunks = splitter.split_documents(documents)
 
#4. Create Gemini embeddings using standard stable model string
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)
 
#5. Store chunks in Chroma
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)
 
#6. Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 4})
 
#7. Define structured output
class RAGAnswer(BaseModel):
    answer: str = Field(
        description="Detailed answer formatted strictly as a numbered list (1., 2., 3.)"
    )
 
    source_snippet: str = Field(
        description="Exact verbatim text excerpt from the PDF context supporting the answer"
    )
 
structured_llm = llm.with_structured_output(RAGAnswer)
 
#8. Create prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert document assistant. Answer thoroughly and in detail "
        "using only the provided context. Format your 'answer' strictly as a numbered list (1., 2., 3., etc.). "
        "For 'source_snippet', provide the exact unedited text excerpt from the document context. "
        "If the answer is not present, say you cannot find it in the document."
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion: {question}"
    )
])

#Helper function to guarantee each numbered point in the answer starts on a new line
def clean_and_format_points(text: str) -> str:
    text = text.strip()
    formatted = re.sub(r'(?<!^)\s+(\d+\.\s+)', r'\n\n\1', text)
    return formatted
 
# 9. Ask questions
print("=" * 70)
print("SMART RAG PDF ASSISTANT READY ")
print("=" * 70)

while True:
    question = input(
        "\nAsk a question about the PDF (type exit to stop): "
    )
 
    if not question.strip():
        continue

    if question.lower() == "exit":
        print("\nRAG application stopped. Goodbye!")
        break
 
    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(question)
 
    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )
    
    #Generate answer
    messages = prompt.invoke({
        "context": context,
        "question": question
    })
 
    response = structured_llm.invoke(messages)
 
    #Professional Terminal Output Layout
    print("\nAnswer -")
    print("")
    formatted_answer = clean_and_format_points(response.answer)
    for line in formatted_answer.split('\n'):
        if line.strip():
            print(f"  {line.strip()}\n")
            
    print("Source Snippet -")
    print("")
    formatted_snippet = clean_and_format_points(response.source_snippet)
    for line in formatted_snippet.split('\n'):
        if line.strip():
            print(f"  {line.strip()}\n")
            
    print("=" * 70)

# import os
# import warnings
# import re
# from pathlib import Path
# from dotenv import load_dotenv
# from pydantic import BaseModel, Field

# # Suppress warnings for a clean terminal output
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=UserWarning)
 
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
 
# # Load .env from the main LLM folder
# ROOT_DIR = Path(__file__).resolve().parent.parent
# load_dotenv(ROOT_DIR / ".env")
 
# # 1. Create Gemini model for text generation
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0
# )
 
# # 2. Load PDF
# pdf_path = ROOT_DIR / "Module2" / "sample_document.pdf"
# print(f"Loading PDF from: {pdf_path}")

# loader = PyPDFLoader(str(pdf_path))
# documents = loader.load()
# print(f"Loaded {len(documents)} pages from the PDF successfully.")

# # 3. Split PDF into intelligent chunks
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len
# )
 
# chunks = splitter.split_documents(documents)
# print(f"Split document into {len(chunks)} optimized chunks.")

# if len(chunks) == 0:
#     raise ValueError("Aborting: No chunks were created because the document text is empty.")
 
# # 4. Create Google Gemini Embeddings
# print("Initializing Google Generative AI embeddings (gemini-embedding-001)...")
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="gemini-embedding-001"
# )
 
# # 5. Store chunks in FAISS vector database (All at once since the PDF is smaller!)
# print("Embedding and storing chunks into FAISS vector database...")
# vector_store = FAISS.from_documents(chunks, embedding=embeddings)
# print("FAISS vector store created successfully!")
 
# # 6. Create retriever
# retriever = vector_store.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 5}
# )
 
# # 7. Define structured output schema
# class RAGAnswer(BaseModel):
#     answer: str = Field(
#         description="Detailed answer formatted strictly as a numbered list (1., 2., 3.)"
#     )
#     source_snippet: str = Field(
#         description="Exact verbatim text excerpt from the PDF context supporting the answer"
#     )
 
# structured_llm = llm.with_structured_output(RAGAnswer)
 
# # 8. Create prompt template
# prompt = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         "You are an expert document assistant. Answer thoroughly and in detail "
#         "using only the provided context. Format your 'answer' strictly as a numbered list (1., 2., 3., etc.). "
#         "For 'source_snippet', provide the exact unedited text excerpt from the document context. "
#         "If the answer is not present, say you cannot find it in the document."
#     ),
#     (
#         "human",
#         "Context:\n{context}\n\nQuestion: {question}"
#     )
# ])

# def clean_and_format_points(text: str) -> str:
#     text = text.strip()
#     formatted = re.sub(r'(?<!^)\s+(\d+\.\s+)', r'\n\n\1', text)
#     return formatted
 
# # 9. Interactive Query Loop
# print("=" * 70)
# print("SMART RAG PDF ASSISTANT READY (FAISS + Gemini Embeddings)")
# print("=" * 70)

# while True:
#     question = input("\nAsk a question about the PDF (type exit to stop): ")
 
#     if not question.strip():
#         continue

#     if question.lower() == "exit":
#         print("\nRAG application stopped. Goodbye!")
#         break
 
#     retrieved_docs = retriever.invoke(question)
#     context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
#     if not context.strip():
#         print("\nAnswer -\n  I cannot find any relevant context in the document for your question.")
#         continue

#     messages = prompt.invoke({
#         "context": context,
#         "question": question
#     })
 
#     response = structured_llm.invoke(messages)
 
#     print("\nAnswer -")
#     print("")
#     formatted_answer = clean_and_format_points(response.answer)
#     for line in formatted_answer.split('\n'):
#         if line.strip():
#             print(f"  {line.strip()}\n")
            
#     print("Source Snippet -")
#     print("")
#     formatted_snippet = clean_and_format_points(response.source_snippet)
#     for line in formatted_snippet.split('\n'):
#         if line.strip():
#             print(f"  {line.strip()}\n")
            
#     print("=" * 70)