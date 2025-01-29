!pip install langgraph langsmith langchain langchain_groq langchain_community
from typing import Annotated
from typing_extensions import TypedDict
!pip install arxiv wikipedia
## Working With Tools

from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun

## Arxiv And Wikipedia tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=300)
arxiv_tool=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300)
wiki_tool=WikipediaQueryRun(api_wrapper=api_wrapper)
wiki_tool.invoke("who is Sharukh Khan")
arxiv_tool.invoke("Attention is all you need")
tools=[wiki_tool]
## Langgraph Application
from langgraph.graph.message import add_messages
class State(TypedDict):
  messages:Annotated[list,add_messages]
from langgraph.graph import StateGraph,START,END
graph_builder= StateGraph(State)
from langchain_groq import ChatGroq
llm=ChatGroq(groq_api_key="",model_name="Gemma2-9b-It")
llm
llm_with_tools=llm.bind_tools(tools=tools)
def chatbot(state:State):
  return {"messages":[llm_with_tools.invoke(state["messages"])]}
from langgraph.prebuilt import ToolNode,tools_condition

graph_builder.add_node("chatbot",chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START,"chatbot")
graph=graph_builder.compile()
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass


user_input=input("")
events=graph.stream(
     {"messages": [("user", user_input)]},stream_mode="values"
)

for event in events:
  event["messages"][-1].pretty_print()
