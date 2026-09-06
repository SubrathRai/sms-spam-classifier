import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()


def transform_text(text ):
    # 1. Convert to lowercase
    text = text.lower()

    # 2. Tokenization
    tokens = nltk.word_tokenize(text)

    # 3. Remove punctuation and special characters
    tokens = [word for word in tokens if word.isalnum()]

    # 4. Remove stopwords
    stop_words = set(stopwords.words('english'))  # set for fast membership checking O(1) compared to lists O(n)
    tokens = [word for word in tokens if word not in stop_words]

    # 5. Stemming
    tokens = [ps.stem(word) for word in tokens]

    return " ".join(tokens)

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))


st.title('Sms Spam Classifier')

input_sms=st.text_input("Enter the message :")
if st.button('Predict'):
    # 1 . Preprocess
    processed_sms=transform_text(input_sms)
    #.2. Vectorize
    vector_input=tfidf.transform([processed_sms])
    # 3. predict
    result=model.predict(vector_input)[0]
    # 4. Display
    if result==1:
        st.header('Spam')
    else:
        st.header('Not Spam')


