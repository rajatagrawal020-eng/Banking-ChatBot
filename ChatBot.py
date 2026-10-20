import json
import os
import random
import ssl
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.title("Banking Chatbot")

# NLTK Download Fallback
try:
  nltk.data.find("tokenizers/punkt")
except LookupError:
  ssl._create_default_https_context = ssl._create_unverified_context
  nltk.download("punkt", quiet=True)

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(BASE_DIR, "intents.json")

with open(json_path, "r", encoding="utf-8") as file:
  intents = json.load(file)

# Training Data Preparation
tags = []
patterns = []
for intent in intents:
  for pattern in intent["patterns"]:
    tags.append(intent["tag"])
    patterns.append(pattern)

vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)


# Chatbot Prediction Function
def chatbot(input_text):
  input_vector = vectorizer.transform([input_text])
  tag = clf.predict(input_vector)[0]
  for intent in intents:
    if intent["tag"] == tag:
      return random.choice(intent["responses"])
  return "Maaf kijiye, main ise samajh nahi paya."


# Session State for Chat History
if "messages" not in st.session_state:
  st.session_state.messages = []

# Previous Messages Render Karein
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# User Input Box (Bottom)
if prompt := st.chat_input("Ask another question..."):
  # User ka message screen par dikhayein
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  # Chatbot response generate karein
  response = chatbot(prompt)

  # Chatbot ka response screen par dikhayein
  st.session_state.messages.append({"role": "assistant", "content": response})
  with st.chat_message("assistant"):
    st.markdown(response)
