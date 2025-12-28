import pickle

# Load model + vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

while True:
    text = input("Enter email message: ")

    # Convert to vector
    vector = tfidf.transform([text])

    # Predict
    result = model.predict(vector)[0]

    if result == 1:
        print("🔥 This is SPAM")
    else:
        print("✔ Not Spam")
