from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
 
load_dotenv()
 
# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)
 
 
# Shared State
class TravelState(TypedDict):
    request: str
    travel: str
    budget: str
    booking: str
    activities: str
    result: str
 
 
# Travel Agent
def travel_agent(state):
    response = llm.invoke(
        "You are a Travel Agent. Create a simple travel plan: "
        + state["request"]
    )
    return {"travel": response.content}
 
 
# Budget Agent
def budget_agent(state):
    response = llm.invoke(
        "You are a Budget Agent. Estimate travel, hotel and food costs: "
        + state["request"]
    )
    return {"budget": response.content}
 
 
# Booking Agent
def booking_agent(state):
    response = llm.invoke(
        "You are a Hotel Booking Agent. Suggest affordable hotels "
        "and booking tips. Do not claim to make reservations: "
        + state["request"]
    )
    return {"booking": response.content}
 
 
# Activities Agent
def activity_agent(state):
    response = llm.invoke(
        "You are an Activity Agent. Suggest tourist places "
        "and activities: " + state["request"]
    )
    return {"activities": response.content}
 
 
# Combine all agent responses
def final_result(state):
    response = llm.invoke(
        "Prepare a complete travel plan using these details:\n"
        + "Travel Plan: " + state["travel"]
        + "\nBudget: " + state["budget"]
        + "\nHotels: " + state["booking"]
        + "\nActivities: " + state["activities"]
    )
    return {"result": response.content}
 
 
# Create Graph
graph = StateGraph(TravelState)
 
# Add agents as nodes
graph.add_node("travel", travel_agent)
graph.add_node("budget", budget_agent)
graph.add_node("booking", booking_agent)
graph.add_node("activities", activity_agent)
graph.add_node("final", final_result)
 
# Connect nodes
graph.add_edge(START, "travel")
graph.add_edge("travel", "budget")
graph.add_edge("budget", "booking")
graph.add_edge("booking", "activities")
graph.add_edge("activities", "final")
graph.add_edge("final", END)
 
# Compile graph
app = graph.compile()
 
 
# Get user input
user_request = input(
    "Enter destination, duration and budget: "
)
 
# Run workflow
result = app.invoke({
    "request": user_request,
    "travel": "",
    "budget": "",
    "booking": "",
    "activities": "",
    "result": ""
})
 
print("\n===== YOUR TRAVEL PLAN =====")
print(result["result"])