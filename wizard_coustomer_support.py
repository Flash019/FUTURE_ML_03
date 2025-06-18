import os
import sys
import torch
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from groq import Groq
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# === CONFIG ===
os.environ["WANDB_DISABLED"] = "true"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
GROQ_API_KEY = "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"
groq_client = Groq(api_key=GROQ_API_KEY)

# === STEP 1: Load and Clean Dataset ===
@st.cache_data
def load_faq_data():
    try:
        df = pd.read_csv("output.csv", sep=None, engine='python')
    except Exception as e:
        st.error(f"❌ Failed to load CSV: {e}")
        return None, None

    if set(['question1', 'question2', 'is_duplicate']).issubset(df.columns):
        df = df.dropna(subset=["question1", "question2", "is_duplicate"])
        df = df[df["is_duplicate"] == 1]
        df = df.rename(columns={"question1": "question", "question2": "answer"})
    elif set(['question', 'answer']).issubset(df.columns):
        df = df[['question', 'answer']].dropna()
    else:
        st.error("❌ Required columns not found.")
        return None, None

    # Add fixed FAQs
    additional_faqs = [
        {"question": "How do I reset my password?", "answer": "Click 'Forgot password' on the login page and follow the instructions."},
        {"question": "What is your return policy?", "answer": "We accept returns within 30 days of purchase."},
        {"question": "How can I track my order?", "answer": "Use the tracking link sent to your email."},
        {"question": "Do you ship internationally?", "answer": "Yes, we ship to most countries."},
        {"question": "How can I contact support?", "answer": "You can contact us via our support page or call 123-456-7890."}
    ]
    df_extra = pd.DataFrame(additional_faqs)
    df = pd.concat([df, df_extra], ignore_index=True)

    return df.to_dict(orient='records'), [item['question'] for item in df.to_dict(orient='records')]

# === STEP 2: Encode FAQs ===
@st.cache_resource
def encode_faqs(faq_questions):
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = []
    batch_size = 32
    for i in tqdm(range(0, len(faq_questions), batch_size), desc="🔄 Embedding"):
        batch = faq_questions[i:i+batch_size]
        encoded = model.encode(batch)
        embeddings.extend(encoded)
    return model, np.array(embeddings)

# === STEP 3: Fallback to Groq ===
def fallback_groq_llama(question):
    try:
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful customer support assistant."},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Groq failed: " + str(e)

# === STEP 4: Get Best Answer ===
def get_best_answer(user_question, embedding_model, faq_data, faq_embeddings, threshold=0.7):
    user_embedding = embedding_model.encode([user_question])
    sims = cosine_similarity(user_embedding, faq_embeddings)[0]
    best_idx = np.argmax(sims)
    confidence = sims[best_idx]
    if confidence > threshold:
        return faq_data[best_idx]["answer"]
    else:
        return fallback_groq_llama(user_question)

# === Streamlit UI ===
def main():
    st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖")
    st.title("🧙‍♂ Wizard Coustomer support 🪄")
    st.markdown("Ask any question below. Here are a few examples:")

    st.info(
        "- How do I reset my password?\n"
        "- What is your return policy?\n"
        "- How can I track my order?\n"
        "- Do you ship internationally?"
    )

    # Load and encode
    with st.spinner("Loading and embedding FAQs..."):
        faq_data, faq_questions = load_faq_data()
        if not faq_data:
            return
        embedding_model, faq_embeddings = encode_faqs(faq_questions)

    # Chat input
    user_question = st.text_input("🔮 Ask your question:")
    if user_question:
        with st.spinner("Thinking..."):
            answer = get_best_answer(user_question, embedding_model, faq_data, faq_embeddings)
            st.success(f"🤖 {answer}")

if __name__ == "__main__":
    main()
