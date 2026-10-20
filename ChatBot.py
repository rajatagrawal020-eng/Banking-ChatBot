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
      response = random.choice(intent["responses"])
      return response
  return "Maaf kijiye, main ise samajh nahi paya."


counter = 0


def main():
  global counter
  st.write("Welcome to the Banking Chatbot!")

  counter += 1
  user_input = st.text_input("You:", key=f"user_input_{counter}")

  if user_input:
    response = chatbot(user_input)
    st.text_area(
        "Chatbot:",
        value=response,
        height=100,
        key=f"chatbot_response_{counter}",
    )

    if response.lower() in ["goodbye", "bye"]:
      st.write("Thanks")
      st.stop()


if __name__ == "__main__":
  main()
