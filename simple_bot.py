from typing import Dict, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    message:list[HumanMessage]


llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def process(state:AgentState) -> AgentState:
    response=llm.invoke(state['message'])
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent=graph.compile()

user_input = input("Enter a message for the agent: ")
while user_input!='exit':
    agent.invoke({"message":[HumanMessage(content=user_input)]})
    user_input=input("Enter a message for the agent (or 'exit' to quit): ")