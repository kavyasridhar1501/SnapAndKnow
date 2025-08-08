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

## 📌 Project Overview
Customer service can be time-consuming and costly for businesses. This chatbot provides:
- Instant responses to customer queries
- Scalable and consistent customer service
- Ability to learn and improve from interactions

Our chatbot uses **Machine Learning** and **pre-trained language models** to understand user intent and respond naturally.  
It supports text-based queries and can be integrated with a website or app.

---

## ✨ Features
- 🗨️ **Real-time Chat** – Engage users with instant responses
- 🧠 **NLP-Powered Responses** – Understands intent & context
- 🌐 **Web Interface** – Easy-to-use frontend for end-users
- ⚡ **Fast Backend Processing** – Built with Node.js & Express
- 📦 **Persistent Storage** – Saves chat history
- 🔧 **Customizable Knowledge Base** – Train with your own data
- 📊 **Analytics Ready** – Track most common queries

---

## 📂 Project Structure
```
customer_chatbot/
│
├── backend/           # API & NLP processing logic
├── frontend/          # User interface (React/Vue)
├── storage/           # Database or JSON storage for chats
├── .env               # Environment variables
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack
**Frontend:** HTML, CSS, JavaScript (React)  
**Backend:** Node.js, Express.js  
**AI/NLP:** Python, Transformers, spaCy / NLTK  
**Database:** JSON / MongoDB  
**Other Tools:** dotenv, axios, socket.io

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/customer_chatbot.git
cd customer_chatbot
```

### 2️⃣ Install Backend Dependencies
```bash
cd backend
npm install
```

### 3️⃣ Install Frontend Dependencies
```bash
cd ../frontend
npm install
```

### 4️⃣ Set up Environment Variables
Create a `.env` file in the `backend/` directory:
```
PORT=5000
MONGO_URI=your_mongodb_uri
OPENAI_API_KEY=your_openai_api_key
```

### 5️⃣ Run the Application
```bash
# Start backend
cd backend
npm start

# Start frontend (in another terminal)
cd frontend
npm start
```

---

## 🚀 Usage
1. Open the frontend in your browser (`http://localhost:3000`)
2. Start typing your questions in the chatbox
3. The bot responds instantly using AI
4. Customize the knowledge base for your use case

---

## 📸 Screenshots

**Home Screen**
![Home Screen](screenshots/home.png)

**Chat in Action**
![Chatbot Conversation](screenshots/chat.png)

---

## 🧠 How It Works
1. **User Query** – User sends a message via the frontend
2. **Backend Processing** – The backend receives the request, processes it with NLP, and calls AI APIs if needed
3. **Response Generation** – AI generates a response based on context and knowledge base
4. **Frontend Display** – The reply is sent back to the frontend and displayed to the user

---

## 🔮 Future Scope
- Add **multi-language support**
- Integrate **voice-based interaction**
- Train a **domain-specific AI model** for better accuracy
- Add **sentiment analysis** to adapt responses
- Implement **chatbot analytics dashboard**
- Integrate with **CRM systems** for advanced customer management

---

💡 *This chatbot can be deployed to any website or internal system to reduce customer service workload and improve customer satisfaction.*
