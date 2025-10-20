# app.py
import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---- Load models & vectorizer ----
lr_model = joblib.load("models/logistic_model.pkl")
svm_model = joblib.load("models/svm_model.pkl")     
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# ---- NLP setup ----
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    # Basic cleaning + lemmatize + remove stopwords and non-alpha tokens
    import re
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(tokens)

# ---- Streamlit UI ----
st.set_page_config(page_title="News Classifier", layout="centered")
st.title("📰 News Category Classifier")
st.write("Enter a news headline or article text. Choose a model and click Classify.")

text_input = st.text_area("Enter your text here", height=200)
model_choice = st.selectbox("Choose model", ("Logistic Regression", "SVM"))

# ---- Class label mapping ----
label_map = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Science"
}

if st.button("Classify"):
    if not text_input.strip():
        st.warning("Please enter some text to classify.")
    else:
        cleaned = preprocess_text(text_input)
        vect = vectorizer.transform([cleaned])
        if model_choice == "Logistic Regression":
            pred = lr_model.predict(vect)[0]
        else:
            pred = svm_model.predict(vect)[0]
        
        # Map numeric prediction to class name
        category = label_map.get(pred, "Unknown")
        
        st.success(f"Prediction: **{category}**")
        st.write("Preprocessed text (what the model sees):")
        st.code(cleaned)