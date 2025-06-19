<p align="center">
  <img src="https://ik.imagekit.io/xojd6puk5/Screenshot%202025-06-19%20133001.png?updatedAt=1750320430929" alt="Wizard Customer Support UI" width="100%">
</p>

<h1 align="center">🧙‍♂️ Wizard Customer Support Chatbot</h1>

<p align="center">
  An intelligent FAQ chatbot using <strong>Sentence-BERT</strong> for question matching and <strong>Groq’s LLaMA 3</strong> for smart fallback answers.
</p>

<p align="center">
  <a href="https://wizardcoustomersupport99.streamlit.app/" target="_blank">🚀 Live Demo</a>
</p>

---

## 🔍 Overview

**Wizard Customer Support** is a Streamlit-based chatbot that handles real-time customer queries using a hybrid of:
- **Semantic Search** with SentenceTransformers
- **Groq LLaMA 3 API** for fallback answers

It’s designed to simulate a helpful, magical support wizard 🧙 that always finds an answer.

---

## 🧰 Tech Stack

- 🔤 **NLP:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- 🤖 **Fallback LLM:** `Groq` + `llama3-8b-8192`
- 📊 **Data:** CSV-based FAQ ingestion
- 🌐 **Frontend:** Streamlit
- 📎 **Similarity:** Cosine similarity for semantic search

---

## 📦 Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/Flash019/FUTURE_ML_03.git
cd FUTURE_ML_03

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
