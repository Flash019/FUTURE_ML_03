<p align="center">
  <img src="https://i.imgur.com/qzWcvGZ.png" alt="Wizard Customer Support Chatbot Banner" width="100%">
</p>

<h1 align="center">🧙‍♂️ Wizard Customer Support Chatbot</h1>

<p align="center">
  An AI-powered FAQ chatbot that matches customer queries to answers using Sentence-BERT and intelligently falls back to Groq’s <strong>LLaMA 3</strong> model for unmatched questions.
</p>

<p align="center">
  <a href="https://wizardcoustomersupport99.streamlit.app/" target="_blank">
    🔗 Live Demo on Streamlit
  </a>
</p>

---

## 🚀 Overview

**Wizard Customer Support** is a real-time chatbot designed to improve customer support with smart FAQ matching and advanced fallback handling using modern AI tools.

- ⚡ Built with **Streamlit** for fast UI
- 🧠 Uses **Sentence Transformers** for semantic search
- 🦙 Falls back to **Groq's LLaMA 3** for unmatched queries
- 📄 Reads from a customizable **FAQ CSV file**
- 🧪 Lightweight, fast, and easy to deploy

---

## 🧰 Tech Stack

- `Streamlit` — UI Framework
- `SentenceTransformers` — for vector embeddings
- `Groq API` — for fallback answers (LLaMA3-8b-8192)
- `Pandas` / `Sklearn` — for data handling and similarity
- `Python` — Core logic

---

## 📂 File Structure


---

## 🧪 How It Works

1. **Load FAQ Dataset** (CSV format)
2. **Embed Questions** using `all-MiniLM-L6-v2`
3. **Find Best Match** using cosine similarity
4. **Fallback to LLaMA 3** if no match is above the threshold

---

## 📦 Installation

```bash
# Clone this repository
git clone https://github.com/your-username/wizard-customer-support.git
cd wizard-customer-support

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
