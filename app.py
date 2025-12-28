import streamlit as st
import pickle

# Load the model and TF-IDF vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# App Title
st.title("📧 Email Spam Detection App")
st.subheader("Using Machine Learning (Naive Bayes + TF-IDF)")

# User input
email_text = st.text_area("Enter the email message below:")

if st.button("Predict"):
    if email_text.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # Transform text
        vector = tfidf.transform([email_text])

        # Prediction
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0][1] * 100

        if prediction == 1:
            st.error("🚨 This is **SPAM**!")
            st.write(f"Spam Probability: **{probability:.2f}%**")
        else:
            st.success("✔ This is **NOT SPAM**")
            st.write(f"Spam Probability: **{probability:.2f}%**")
