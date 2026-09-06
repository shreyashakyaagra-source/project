import nltk
from nltk.corpus import stopwords
import contractions as contract
import re
import streamlit as st

nltk.download('stopwords')

stop_words = set(stopwords.words ('english'))

def clean_text(text) :
    text = text .lower()

    text = re.sub(r"[^a-zA-z\s]" ,'',text)

    text= contract.fix(text)

    text= re.sub(r'<.*?>',"",text)

    text=re.sub(r'\s+',' ',text)

    text= re.sub(r"https \S+|www\S+","",text)

    return text

def remove_stopwords(text):
    words=text.lower().split()
    filtered_words=[word for word in words if word not in stop_words]
    return words, filtered_words

st.set_page_config(
    page_title="NLP text cleaner",
    page_icon="🧠",
    layout="wide"
)

st.title("NLP text cleaning and stopwords removal app")
st.write("Enter raw text below to clean it and remove stopwords")

user_input=st.text_area(
    "Nter your text here:"
)

if st.button("Process text"):
    cleaned_text=clean_text(user_input)

    original_words, filtered_words= remove_stopwords(cleaned_text)

    st.subheader("original Text")
    st.write(user_input)

    st.subheader("cleaned Text")
    st.write(cleaned_text)

    st.subheader("words before removing stopwords")
    st.write(original_words)

    st.subheader("word after removing stopwords")
    st.write(filtered_words)

