import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training model jika belum ada
if not os.path.exists("model.pkl"):

    df = pd.read_csv(
        "SMSSpamCollection",
        sep="\t",
        names=["label", "message"]
    )

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df["message"])
    y = df["label"]

    model = MultinomialNB()
    model.fit(X, y)

    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📩 SMS Spam Detection")

pesan = st.text_area("Masukkan Pesan SMS")

if st.button("Deteksi"):

    data = vectorizer.transform([pesan])

    hasil = model.predict(data)[0]

    if hasil == "spam":
        st.error("🚨 SPAM")
    else:
        st.success("✅ HAM (Bukan Spam)")