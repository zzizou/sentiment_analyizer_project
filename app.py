from flask import Flask, render_template, request, url_for
import numpy as np
import pickle 
from sklearn.feature_extraction.text import TfidfVectorizer
app = Flask(__name__)
import pandas as pd
import os

model = pickle.load(open("hatespeech_model.pkl", "rb"))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl' , 'rb'))

@app.route('/')
def hell_world():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def result():
    new_text = request.form['user_input']
    print("New Text String" , new_text)
    test_data = [new_text]
    print("Test Data List" , test_data)
    # # Preprocess the user input (tokenize and vectorize)
    new_text_tfidf = tfidf_vectorizer.transform(test_data)
    label = model.predict(new_text_tfidf)
    print("Label is" , label)
    if label == 0.0:
        resData = "NO HATE"
    elif label == 1.0:
        resData ="HATE COMMENT!" 
    elif label == 2.0:
        resData = "OFFENSIVE"
        
    # Process the model's prediction
    print("RESULT IS" , resData)
    return render_template('index.html' , result=resData )


if __name__ == '__main__':
    app.run()

