# AG News - Multiclass Text Classification (NLP)

This project classifies news articles into one of four categories — **World**, **Sports**, **Business**, or **Science** — using a machine learning model trained on the AG News dataset.  
The model uses **Scikit-learn**, **NLTK**, and **Streamlit** for deployment.

---

## Live App

👉 **Try it here:** [https://agnewsmulticlassclassificationnlp.streamlit.app/](https://agnewsmulticlassclassificationnlp.streamlit.app/)

---

## Project Overview

This project demonstrates a simple end-to-end NLP pipeline:
1. **Text Preprocessing** – Tokenization, stopword removal, and text cleaning using NLTK.  
2. **Feature Extraction** – TF-IDF vectorization.  
3. **Model Training** – A Scikit-learn classifier trained on the AG News dataset.  
4. **Deployment** – Interactive Streamlit web app for classifying new text inputs.

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** – for the web app  
- **Scikit-learn** – for model training & prediction  
- **NLTK** – for natural language preprocessing  
- **Joblib** – for model serialization  

---

## Installation & Setup

To run the project locally:

```bash
# 1. Clone the repository
git clone https://github.com/malakasaber/nlp-practice-projects.git
cd "Elevvo Pathways Projects/AG News - Multiclass Classification/Deployment"

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate    # On Windows use: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
