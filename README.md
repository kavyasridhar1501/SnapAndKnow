# Snap&Know – A Visual Product Discovery Assistant

*See a product online or in person, snap a photo, and get answers—brand, price, and what people think—straight from the picture.*    

This project accepts an image or a text query and returns key details, including brand/model, colour, price, and user reviews.  
It fuses image understanding (captioning, OCR, colour detection) with retrieval-augmented generation(RAG) over a review corpus, adds targeted web enrichment when available, and synthesises a clear response via an LLM.

## Technical Highlights

- **Vision pipeline (Captioning + OCR + Color)**
  - Extracts important information from images like text (brand/model), colour, etc., turning pictures into structured hints.
- **RAG over product reviews**
  - Uses a disk-persisted vector index to retrieve relevant user reviews.
- **Targeted enrichment**
  - Price and metadata lookups using OCR-derived cues.
- **LLM synthesis**
  - Generates final answers by fusing responses from the Vision Pipeline, RAG, and Enrichment into a single, readable response.
- **LangChain Tools & Routing**
  - Wraps core capabilities and routes queries to the right toolset based on user intent and available context.
- **LangGraph Stateful Workflow**
  - Orchestrates multi-step flows and preserves state (latest image, intermediate results) across turns.

---

## Project Structure
```
SnapAndKnow/
├─ README.md
├─ .gitignore
├─ .env                        
├─ requirements.txt
├─ .github/
│  └─ workflows/
│     └─ pages.yml             # GitHub Pages deploy for /frontend
│
├─ storage/
│  ├─ default__vector_store.json
│  └─ docstore.json
│     ├─ graph_store.json
│     ├─ image__vector_store.json
│     └─ index_store.json
│
├─ frontend/
│  ├─ index.html               
│  └─ icons/
│     ├─ chat_bot.png
│     ├─ upload.png
│     └─ question.png
│
└─ backend/
   ├─ app.py                   # Flask app
   ├─ agent.py                 # LangChain AgentExecutor + tools + memory
   ├─ image_pipeline.py        # Vision Pipeline
   ├─ llm_wrapper.py           # Wraps GROQ LLM + Loads LlamaIndex RAG from ./storage
   ├─ langchain_utils.py       # Prompt templates + helper chains
   ├─ enrichment.py            # “Live Lookup” helpers (brand/model/price)
   ├─ rag_setup.py             # Index builder (Streams reviews to ./storage)

```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/kavyasridhar1501/SnapAndKnow.git
cd SnapAndKnow
```

### 2. Install All Dependencies
```bash
pip install -U pip
pip install -r requirements.txt
```

### 3. Set up Environment Variables
Create a `.env` file in the project directory:
```
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
GROQ_API_KEY=your_groq_api_key
FLASK_SECRET_KEY=your_flask__key
```

### 4. Run the rag_setup.py
```bash
python3 backend/rag_setup.py
```

### 4. Run the application
```bash
python3 backend/app.py
```
pen the frontend in your browser (`http://127.0.0.1:8000/`)

---

## Front-End 
[View Static Front-End]([https://your-username.github.io/your-repo/](https://kavyasridhar1501.github.io/SnapAndKnow/))


## Future Scope
- Add **multi-language support**
- Integrate **voice-based interaction**
- Train a **domain-specific AI model** for better accuracy
- Add **sentiment analysis** to adapt responses
- Implement **chatbot analytics dashboard**
- Integrate with **CRM systems** for advanced customer management

---
