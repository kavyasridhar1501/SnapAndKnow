import os
import json
import itertools
from dotenv import load_dotenv
from datasets import load_dataset, logging

logging.set_verbosity_error()

# Load GROQ_API_KEY from .env
load_dotenv()                  

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Please set GROQ_API_KEY in your .env")

embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")
llm_model   = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)

# Stream and Collecting upto MAX_PER_SPLIT reviews/category
MAX_PER_SPLIT = 5000
REVIEW_CFGS = [
    "raw_review_All_Beauty",
    "raw_review_Toys_and_Games",
    "raw_review_Cell_Phones_and_Accessories",
    "raw_review_Industrial_and_Scientific",
    "raw_review_Gift_Cards",
    "raw_review_Musical_Instruments",
    "raw_review_Electronics",
    "raw_review_Handmade_Products",
    "raw_review_Arts_Crafts_and_Sewing",
    "raw_review_Baby_Products",
    "raw_review_Health_and_Household",
    "raw_review_Office_Products",
    "raw_review_Digital_Music",
    "raw_review_Grocery_and_Gourmet_Food",
    "raw_review_Sports_and_Outdoors",
    "raw_review_Home_and_Kitchen",
    "raw_review_Subscription_Boxes",
    "raw_review_Tools_and_Home_Improvement",
    "raw_review_Pet_Supplies",
    "raw_review_Video_Games",
    "raw_review_Kindle_Store",
    "raw_review_Clothing_Shoes_and_Jewelry",
    "raw_review_Patio_Lawn_and_Garden",
    "raw_review_Unknown",
    "raw_review_Books",
    "raw_review_Automotive",
    "raw_review_CDs_and_Vinyl",
    "raw_review_Beauty_and_Personal_Care",
    "raw_review_Amazon_Fashion",
    "raw_review_Magazine_Subscriptions",
    "raw_review_Software",
    "raw_review_Health_and_Personal_Care",
    "raw_review_Appliances",
    "raw_review_Movies_and_TV",
]

raw_texts = []
print("Streaming review splits:")
for cfg in REVIEW_CFGS:
    print(f"\n->{cfg}")
    try:
        ds = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            cfg,
            split="full",
            streaming=True,
        )
    except Exception as e:
        print("", e)
        continue

    kept = 0
    for rec in itertools.islice(ds, MAX_PER_SPLIT):
        text = rec.get("text") or rec.get("review_body") or ""
        if not text.strip():
            continue
        payload = {
            "type":   "review",
            "asin":   rec.get("parent_asin") or rec.get("asin", ""),
            "rating": rec.get("rating") or rec.get("overall"),
            "title":  rec.get("title") or rec.get("review_title") or "",
            "text":   text,
        }
        raw_texts.append(json.dumps(payload, ensure_ascii=False))
        kept += 1
    print(f"Collected {kept}")

print(f"\nTotal docs collected = {len(raw_texts)}")


# Wrap into Llama-index Documents
docs = [Document(text=t) for t in raw_texts]

print("\nBuilding the VectorStoreIndex")
storage_context = StorageContext.from_defaults() 
index = VectorStoreIndex.from_documents(
    docs,
    embed_model=embed_model,
    llm=llm_model,
    storage_context=storage_context,
    show_progress=True,
)

print("Persisting into ./storage…")
storage_context.persist()

# Verifying RAG Output
print("\n./storage now contains:")
for fn in sorted(os.listdir("storage")):
    print("   ", fn)
print("\nRAG setup complete.")
