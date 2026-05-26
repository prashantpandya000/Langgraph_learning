import os
from typing import TypedDict,List,Dict,Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class AgentState(TypedDict):
    message: List[Union[HumanMessage, SystemMessage, AIMessage]]


def process(state:AgentState) -> AgentState:
    """Process the agent's state by invoking the language model and appending the response to the message history."""
    response=llm.invoke(state['message'])
    print(f"\nAI: {response.content}")
    state['message'].append(AIMessage(content=response.content))
    print("current state:", state)
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent=graph.compile()

conversation_history: List[Union[HumanMessage, SystemMessage, AIMessage]] = []

user_input = input("Enter a message for the agent: ")
while user_input!='exit':
    conversation_history.append(HumanMessage(content=user_input))
    result=agent.invoke({"message": conversation_history})
    conversation_history=result['message']
    user_input=input("Enter a message for the agent (or 'exit' to quit): ")


with open("logging.txt", "w") as file:
    file.write("Conversation History:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"Human: {message.content}\n")
        elif isinstance(message, SystemMessage):
            file.write(f"System: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n")

print("Conversation history has been logged to logging.txt")