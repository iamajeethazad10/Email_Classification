# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:35:43 2023

@author: Ajeeth A K Azad
"""


import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import re 
from nltk.tokenize import word_tokenize
import nltk
from sklearn.svm import SVC
import joblib
import numpy as np
from PIL import Image




from nltk.corpus import stopwords


custom_stopwords = set(stopwords.words('english'))

img = Image.open(r'C:\Users\god\Desktop\Internship_Project2\bg.jpg') #Image name
#Coordinates
left = 0
top = 0
right =0
bottom = 0

def text_cleaner(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\w*\d\w*', ' ', text)
    text = re.sub('[0-9]+', ' ', text)
    text = re.sub('[\'"“”…]', ' ', text)
    text = re.sub('[\n]', ' ', text)
    text = re.sub('[\s]+', ' ', text)  
    text = text.lower()
    text = word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = ' '.join(y)

    y = []
    for i in text.split():
        if i not in custom_stopwords and i not in string.punctuation:
            y.append(i)

    return ' '.join(y)




with open('vector.pkl', 'rb') as vector_file, open('model.pkl', 'rb') as model_file:
    tfidf = pickle.load(vector_file)
    model = pickle.load(model_file)

st.image(img,width=500,use_column_width=True)

st.title("Email Classifier")

input_sms = st.text_input("Enter the Content")




if st.button("Predict"):
    transformed_sms = text_cleaner(input_sms)
    
    # Wrap transformed_sms in a list
    vector_input = tfidf.transform([transformed_sms])
    
    # Convert sparse matrix to dense array
    dense_vector_input = vector_input.toarray()
    
    result = model.predict(dense_vector_input)[0]
    
    if result == 0:
       st.markdown("<h1 style='color: Yellow;'>Abusive</h1>", unsafe_allow_html=True)
    else:
       st.markdown("<h1 style='color: Blue;'>Not Abusive</h1>", unsafe_allow_html=True)

       
        



  
    


    



