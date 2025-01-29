!pip install -q --upgrade langchain langchain_community langgraph requests beautifulsoup4 googlesearch-python newspaper3k anthropic

from langchain.agents import Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from anthropic import Client, HUMAN_PROMPT, AI_PROMPT
import os
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from newspaper import Article
from typing import TypedDict, Annotated

# Set up the Claude API (Anthropic)
os.environ["ANTHROPIC_API_KEY"] = ""  # Replace with your actual key
client = Client(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Web scraping function
def get_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# Web search function
def web_search(query):
    try:
        search_results = search(query, num_results=3)
        result_texts = []

        for url in search_results:
            page = requests.get(url)
            soup = BeautifulSoup(page.content, 'html.parser')
            text = " ".join([p.text for p in soup.find_all('p')])
            
            if len(text.split()) < 30:  # Fallback to Newspaper if result text is too short
                text = get_article_text(url)
            
            result_texts.append(text)

        return "\n\n".join(result_texts)
    
    except Exception as e:
        return f"Error during web search: {str(e)}"

# Claude API call
def claude_respond(state: State):
    user_input = state["messages"][-1].content
    if not user_input:
        return {"messages": state["messages"] + [AIMessage(content="Please provide a query.")]}

    try:
        response = client.completions.create(
            prompt=f"{HUMAN_PROMPT}{user_input}{AI_PROMPT}",
            stop_sequences=[HUMAN_PROMPT],
            max_tokens_to_sample=300,
            model="claude-2",
            temperature=0.7,
        )
        completion_text = response.completion
        return {"messages": state["messages"] + [AIMessage(content=completion_text)]}
    
    except Exception as e:
        return {"messages": state["messages"] + [AIMessage(content=f"Error: {str(e)}")]}

# Define LangGraph State
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# Add nodes and edges to the graph
graph_builder.add_node("claude", claude_respond)
wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki_tool = Tool(
    name="Wikipedia",
    func=wiki_api_wrapper.run,
    description="Retrieve information from Wikipedia.",
)

search_tool = Tool(
    name="WebSearch",
    func=web_search,
    description="Perform a general web search for additional information.",
)

tool_node = ToolNode(tools=[wiki_tool, search_tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "claude")
graph_builder.add_edge("claude", "tools")
graph_builder.add_edge("tools", END)

graph = graph_builder.compile()

# Main function
def chatbot():
    print("Welcome to the chatbot! Type 'exit' to quit.")
    
    state = {"messages": []}
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        state["messages"].append(HumanMessage(content=user_input))
        
        events = graph.stream(state, stream_mode="values")
        
        for event in events:
            last_message = event["messages"][-1]
            if isinstance(last_message, AIMessage):
                print(f"Bot: {last_message.content}")
                break

if __name__ == "__main__":
    chatbot()
