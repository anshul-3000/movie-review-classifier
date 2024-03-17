import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
import sklearn
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


def transform(text):
    text = text.lower()  # converting into lowercase
    text = nltk.word_tokenize(text)  # converting into tokens

    new = []
    for i in text:  # eliminating special characters
        if i.isalnum():
            new.append(i)

    text = new[:]  # cloning the data becoz list is mutable
    new.clear()
    for i in text:  # eliminating stopletters & punctuations
        if i not in stopwords.words('english') and i not in string.punctuation:
            new.append(i)

    text = new[:]
    new.clear()
    for i in text:
        new.append(ps.stem(i))

    return " ".join(new)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
st.title("Positive/Negative review classifier")
input_review = st.text_area("Enter the review")
if st.button('Predict'):
    # 1. Preprocess
    transformed_review = transform((input_review))
    # 2. Vectorize
    vector_review = tfidf.transform([transformed_review])
    # 3. Predict
    result = model.predict(vector_review)[0]
    # 4. Display
    if result == 1:
        st.header("Positive Review")
    else:
        st.header("Negative Review")

