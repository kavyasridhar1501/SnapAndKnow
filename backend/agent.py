import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
import image_pipeline
from langchain_utils import make_conv_chain 

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in environment (.env)")

def describe_image(prompt: str) -> str:
    # Describes the latest image with a short caption.
    if image_pipeline.last_image is None:
        return "No image was uploaded."
    return image_pipeline.image_blurb(image_pipeline.last_image, prompt or "Describe this image.")

def detect_color(_: str) -> str:
    # Returns the dominant color of the last uploaded image.
    if image_pipeline.last_image is None:
        return "No image was uploaded."
    return image_pipeline.get_dominant_color(image_pipeline.last_image)


_conv_chain = make_conv_chain()

def rag_answer(query: str) -> str:
    return _conv_chain.predict(question=query)

tools = [
    Tool(name="DescribeImage", func=describe_image,
         description="Use when the user asks what an uploaded image is or to describe it."),
    Tool(name="DetectColor", func=detect_color,
         description="Use when the user asks about the color of the uploaded image."),
    Tool(name="RAGAnswer", func=rag_answer,
         description="Use for factual/product questions over the reviews corpus."),
]


llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-70b-8192", temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

prefix = (
    "You are a helpful shopping assistant. "
    "You have tools: DescribeImage, DetectColor, and RAGAnswer. "
    "If there is an uploaded image and the user asks 'what is this', "
    "first call DescribeImage. If the user asks about color of an image, "
    "call DetectColor. Otherwise use RAGAnswer. Keep replies concise."
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,     
    agent_kwargs={"prefix": prefix},
)
