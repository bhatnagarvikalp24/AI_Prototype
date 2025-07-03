# 🧠 Agentic AI Assistant for Insurance Analytics

An interactive Streamlit app built with LangGraph, Vanna.AI, SerpAPI, and OpenAI. This intelligent assistant can:

- Auto-route queries to SQL, Google Search, or Document updates
- Generate and execute SQL using Vanna.AI
- Perform insurance-specific Google searches via SerpAPI
- Update Word documents programmatically using AI + fuzzy matching

---

## 🚀 Features

- 🔀 **Dynamic Routing** with LangGraph (SQL / Search / Document)
- 📊 **SQL Generation** using Vanna + LLM (e.g., GPT-3.5)
- 🌐 **Insurance-focused Google Search** via SerpAPI
- 📄 **Intelligent Document Updates** (e.g., update tables in .docx)
- 🧠 LLM-powered decisions using OpenAI (GPT-3.5 or GPT-4)

---

## 📦 Requirements

- Python 3.8+
- [OpenAI account](https://platform.openai.com/)
- [Vanna.AI API Key](https://vanna.ai/)
- [SerpAPI account](https://serpapi.com/)

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/insurance-agentic-assistant.git
cd insurance-agentic-assistant
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt

OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
VANNA_API_KEY=your_vanna_api_key
VANNA_MODEL_NAME=vanna-openai
SERPAPI_API_KEY=your_serpapi_key

streamlit run main.py

🧱 Architecture
LangGraph: routes logic to appropriate node

Vanna.AI: turns natural language into SQL (with your DB)

SerpAPI: performs Google search restricted to insurance topics

OpenAI: powers routing and document table recognition

Streamlit: UI layer for interactivity

├── main.py                 # Streamlit + LangGraph agent logic
├── .env                    # API keys
├── requirements.txt
├── updated_doc.docx        # Auto-generated if doc updated


├── main.py                 # Streamlit + LangGraph agent logic
├── .env                    # API keys
├── requirements.txt
├── updated_doc.docx        # Auto-generated if doc updated


📬 Contact
Made by Vikalp Bhatnagar (bhatnagarvikalp24@gmail.com).
Open to contributions, improvements, and domain expansion (e.g., banking, legal).
