 
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
 
 
# 1. Define State
class SupportState(TypedDict):
    question: str
    category: str
    response: str
 
 
# 2. Create Nodes
 
def classify_question(state: SupportState):
 
    question = state["question"].lower()
 
    if "order" in question:
        return {"category": "order"}
 
    elif "refund" in question:
        return {"category": "refund"}
 
    else:
        return {"category": "general"}
 
 
def order_node(state: SupportState):
 
    return {
        "response": "Please check your order tracking details."
    }
 
 
def refund_node(state: SupportState):
 
    return {
        "response": "Please submit your refund request to customer support."
    }
 
 
def general_node(state: SupportState):
 
    return {
        "response": "Hello! How can I help you today?"
    }
 
 
# 3. Create StateGraph
graph = StateGraph(SupportState)
 
 
# 4. Add Nodes
graph.add_node("classify", classify_question)
graph.add_node("order", order_node)
graph.add_node("refund", refund_node)
graph.add_node("general", general_node)
 
 
# 5. Create Conditional Routing
def check_category(state: SupportState):
 
    if state["category"] == "order":
        return "order"
 
    elif state["category"] == "refund":
        return "refund"
 
    else:
        return "general"
 
 
# 6. Add Edges
graph.add_edge(START, "classify")
 
graph.add_conditional_edges(
    "classify",
    check_category
)
 
graph.add_edge("order", END)
graph.add_edge("refund", END)
graph.add_edge("general", END)
 
 
# 7. Compile Graph
app = graph.compile()
 
 
# 8. Execute Workflow
while True:
 
    question = input("\nEnter your question (type exit to stop): ")
 
    if question.lower() == "exit":
        print("Application stopped.")
        break
 
    result = app.invoke({
        "question": question,
        "category": "",
        "response": ""
    })
 
    print("\nCategory:", result["category"])
    print("Response:", result["response"])
 
 """
The 4 Building Blocks of Your Workflow
Before looking at the journey, here are the four core pieces making it happen:

1. The Case Folder (SupportState):
   Every time a customer sends a message, the system opens a digital folder. As the
   message moves through the system, each step writes its notes into this folder:
   - What did the customer ask?
   - What category is it? (Order, Refund, or General)
   - What is the drafted answer?
   - Did a human manager approve it?
   - Which steps have been completed so far?

2. The Specialist Desks (Nodes):
   Instead of one worker doing everything, you have 5 focused workers, each with one
   exact job:
   - Desk 1 — Triage / Classify: Reads the message and decides which department should
     handle it.
   - Desk 2 — Order Specialist: Handles shipping, tracking, and delivery questions.
   - Desk 3 — Refund Prep Specialist: Reads a refund request and writes a
     summary/recommendation for a manager to review.
   - Desk 4 — Refund Execution Specialist: Sends the final "Approved" or "Declined"
     message only after a human signs off.
   - Desk 5 — General Support: Handles everyday questions (like working hours).

3. The Traffic Controller (Conditional Routing):
   Right after Desk 1 labels the folder (Order, Refund, or General), the Traffic
   Controller looks at that label and sends the folder down only one path. It never
   wastes time visiting irrelevant desks.

4. The Filing Cabinet (Checkpointing & Thread ID):
   Every conversation has a unique tracking number called a thread_id. After every
   single desk finishes its job, the system takes a photocopy of the Case Folder and
   saves it in the Filing Cabinet. Because of this:
   - The system remembers how many turns have happened in that thread.
   - If the system has to stop and wait for a human, it doesn't forget what it was
     doing.

Step-by-Step: How the Workflow Runs
Depending on what the user types, the Case Folder travels down one of two paths:

Path 1: The Fast Track (Low-Risk Questions: Order or General)
Example: "Where is my order #ORD-4821?" or "What are your working hours?"
1. Start: You type a message in the Streamlit webpage and hit Enter. The webpage sends
   your message and your thread_id to the backend.
2. Triage (classify node): The AI reads your question and tags the Case Folder as
   ORDER (or GENERAL). It also marks requires_approval = False because answering a
   question carries zero financial risk.
3. Routing: The Traffic Controller sees the ORDER tag and routes the folder straight
   to the Order Specialist.
4. Drafting & Finishing (order node): The AI writes a helpful reply about order
   tracking and puts it in the Case Folder as the final answer.
5. Save & Deliver: The Filing Cabinet saves a snapshot (Checkpoint Turn #1), and the
   answer appears on your screen along with the route badge and execution trace.
   Done.

Path 2: The Approval Track (High-Risk Actions: Refund)
Example: "I want a refund of $89 for a damaged headphone order."
Because giving money back involves financial risk, you programmed a Human-in-the-Loop
(interrupt_before) rule: never let the AI finalize a refund alone.
1. Start: You send the refund request.
2. Triage (classify node): The AI reads the message, tags the folder as REFUND, and
   flags requires_approval = True.
3. Routing: The Traffic Controller sees REFUND and sends the folder to the Refund Prep
   Specialist.
4. Preparation (refund_prepare node): The AI writes a short summary of the refund
   request so a human supervisor doesn't have to read everything from scratch.
5. THE MANDATORY PAUSE (Human-in-the-Loop Gate):
   - Normally, the next step is refund_execute.
   - However, right before entering that desk, the system hits a red stop sign.
   - The Filing Cabinet saves the exact state of the Case Folder under your thread_id
     and stops running.
   - On your screen, the chat input locks, a warning banner appears showing the AI's
     draft summary, and two buttons appear: Approve & Resume Graph and Reject
     Refund.
6. Human Decision:
   - A human supervisor reviews the draft, optionally types a note (e.g., "Verified
     return tracking"), and clicks Approve or Reject.
7. Resuming from the Checkpoint:
   - Clicking the button sends the human's decision (approved = True/False + the
     supervisor's note) back to the backend using the exact same thread_id.
   - LangGraph pulls the paused Case Folder out of the Filing Cabinet, updates it with
     the supervisor's decision, and wakes up the workflow right where it stopped.
8. Final Execution (refund_execute node):
   - If Approved, this desk generates the official refund confirmation message
     (including the supervisor's note).
   - If Rejected, it generates a polite decline message with the supervisor's reason.
9. Save & Deliver: The Filing Cabinet saves the completed state, unlocks the chat bar,
   and shows the final resolution on your screen. Done.
"""