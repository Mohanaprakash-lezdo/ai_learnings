# Install required packages
!pip install -q -U langchain langchain-community langgraph beautifulsoup4 requests

# Import necessary libraries
import os
from langchain.llms import Cohere
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langgraph.graph import StateGraph, END
from bs4 import BeautifulSoup
import requests
from typing import List, Union, TypedDict
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage

# Set API keys
os.environ["COHERE_API_KEY"] = "your_valid_cohere_api_key_here"  # Replace with your valid API key

# Define web scraping function
def web_scraper_tool(query):
    print(f"Searching for: {query}")
    search_url = f"https://www.google.com/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        snippets = soup.select("div.BNeawe.s3v9rd.AP7Wnd")
        if snippets:
            return snippets[0].get_text()
        else:
            return "No relevant information found on the web."
    else:
        return "Failed to fetch search results."

# Define the scraping tool
scraping_tool = Tool(
    name="WebScraper",
    func=web_scraper_tool,
    description="Fetches search results and extracts useful information.",
)

# Initialize the Cohere LLM
llm = Cohere(model="command-xlarge")

# Combine tools into an agent
tools = [scraping_tool]

agent_runable = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# Define agent state for `langgraph`
class AgentState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: List[tuple[AgentAction, str]]

# Define agent execution logic
def run_agent(data):
    agent_outcome = agent_runable.invoke(data["input"])
    return {"agent_outcome": agent_outcome}

# Define tool execution logic
def execute_tools(data):
    agent_action = data["agent_outcome"]
    if isinstance(agent_action, AgentAction):
        tool_name = agent_action.tool
        tool_input = agent_action.tool_input
        for tool in tools:
            if tool.name == tool_name:
                if isinstance(tool_input, str):  # Ensure only strings are passed
                    output = tool.func(tool_input)
                    data["intermediate_steps"] = data.get("intermediate_steps", []) + [(agent_action, str(output))]
                    return {"intermediate_steps": data["intermediate_steps"]}
                else:
                    raise ValueError(f"Tool '{tool_name}' received invalid input: {tool_input}")
    return {"intermediate_steps": data.get("intermediate_steps", [])}

# Define conditional edge logic
def should_continue(data):
    if isinstance(data["agent_outcome"], AgentFinish):
        return "end"
    return "continue"

# Define the workflow graph
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)

# Set entry point
workflow.set_entry_point("agent")

# Add conditional edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "action", "end": END},
)
workflow.add_edge("action", "agent")

# Compile the workflow graph
app = workflow.compile()

# Interactive User Input
print("Agent initialized! Type 'exit' to quit.")
chat_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    input_data = {"input": user_input, "chat_history": chat_history}
    try:
        for state in app.stream(input_data):
            chat_history = state.get("chat_history", [])
            response = state.get("agent_outcome", "")
            if isinstance(response, AgentFinish):
                print(f"Agent: {response.return_values['output']}")
    except Exception as e:
        print(f"Error: {e}")
