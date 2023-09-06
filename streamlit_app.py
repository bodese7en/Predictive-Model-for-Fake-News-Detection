import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
import nltk

# Download NLTK stopwords (only need to do this once)
nltk.download('stopwords')

# Load English stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    return text

def output_label(n):
    if n == 0:
        return "True News"
    elif n == 1:
        return "Fake News"

def predict_fake_news(news, model, vectorizer):
    news = preprocess_text(news)  # Preprocess the input news
    news_list = [news]
    news_vector = vectorizer.transform(news_list).toarray()
    prediction = model.predict(news_vector)
    return output_label(prediction[0])

# Load the TF-IDF vectorizer and model
tfidf = joblib.load(open('vectorizer.pkl', 'rb'))
model = joblib.load(open('model.pkl', 'rb'))

st.title('Fake News Detection')

# Input for the news article
news_article = st.text_area('Enter the news article:', '')

if st.button('Predict'):
    if news_article:
        prediction = predict_fake_news(news_article, model, tfidf)
        st.header(prediction)
    else:
        st.warning('Please enter a news article for prediction')
