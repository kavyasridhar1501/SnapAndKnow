import os
from typing import Optional, List
from dotenv import load_dotenv
from langchain.llms.base import LLM

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in environment (.env)")

from llama_index.core.settings import Settings         
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model
Settings.llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
Settings.num_output = 512   
Settings.chunk_size = 1024  

# Load the persisted index
STORAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "storage"))
if not os.path.isdir(STORAGE_DIR):
    raise FileNotFoundError(
        f"No persisted index found at {STORAGE_DIR}. Run rag_setup.py first."
    )

storage_ctx = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
_index = load_index_from_storage(storage_ctx)

# Building a single cached query engine
_query_engine = _index.as_query_engine(similarity_top_k=5)

# LangChain LLM wrapper that answers by querying the persisted LlamaIndex.
class GroqLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "groq-index-wrapper"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        result = _query_engine.query(prompt)
        return getattr(result, "response", str(result))

    @property
    def _identifying_params(self) -> dict:
        return {"model": "llama3-70b-8192"}
