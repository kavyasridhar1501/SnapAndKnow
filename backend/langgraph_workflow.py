from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
import image_pipeline
from langchain_utils import make_conv_chain

# Conversation state schema
class ChatState(TypedDict):
    last_image: object | None  
    query: str                 

# Prebuilding RAG + memory chain
conv_chain = make_conv_chain()

# Initiate the graph with the schema
graph = StateGraph(state_schema=ChatState)

# Node functions
def start(state: ChatState):
    return None

def has_image(state: ChatState) -> bool:
    return state.get("last_image") is not None

def wants_color(state: ChatState) -> bool:
    return "color" in state.get("query", "").lower()

def describe_image(state: ChatState) -> str:
    return image_pipeline.image_blurb(
        state["last_image"],
        "Please describe the contents of this image."
    )

def detect_color(state: ChatState) -> str:
    return image_pipeline.get_dominant_color(state["last_image"])

def rag_answer(state: ChatState) -> str:
    return conv_chain.predict(question=state["query"])


# Adding nodes to the graph
start_node    = graph.add_node("Start",       ToolNode([start]))
describe_node = graph.add_node("DescribeImage", ToolNode([describe_image]))
detect_node   = graph.add_node("DetectColor", ToolNode([detect_color]))
rag_node      = graph.add_node("RAGAnswer",   ToolNode([rag_answer]))


# Routing via conditional edges

# Image & color request → DetectColor
graph.add_conditional_edges(
    start_node,
    detect_node,
    lambda s: has_image(s) and wants_color(s)
)

# Image but not color → DescribeImage
graph.add_conditional_edges(
    start_node,
    describe_node,
    lambda s: has_image(s) and not wants_color(s)
)

# No image → RAGAnswer directly
graph.add_conditional_edges(
    start_node,
    rag_node,
    lambda s: not has_image(s)
)

graph.add_edge(describe_node, rag_node)

# Compile into executable agent
workflow_agent = graph.compile(entry_node=start_node)
