import json
import os
import random
import ssl
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.title("Banking Chatbot")

try:
  nltk.data.find("tokenizers/punkt")
except LookupError:
  ssl._create_default_https_context = ssl._create_unverified_context
  nltk.download("punkt", quiet=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(BASE_DIR, "intents.json")

with open(json_path, "r", encoding="utf-8") as file:
  intents = json.load(file)

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

def chatbot(input_text):
  input_vector = vectorizer.transform([input_text])
  tag = clf.predict(input_vector)[0]
  for intent in intents:
    if intent["tag"] == tag:
      return random.choice(intent["responses"])
  return "Sorry!. I could not understand."

if "messages" not in st.session_state:
  st.session_state.messages = []

for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

if prompt := st.chat_input("Ask another question..."):
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  response = chatbot(prompt)

  st.session_state.messages.append({"role": "assistant", "content": response})
  with st.chat_message("assistant"):
    st.markdown(response)
