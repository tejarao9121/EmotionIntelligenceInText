import streamlit as st 
from  sklearn.metrics import accuracy_score
import pickle
from compress_pickle import dump,load
import time

st.title('Twitter Sentiment Analysis')
model =pickle.load(open('twitter_sentiment.pkl','rb'))

tweet=st.text_input("Enter you tweet")

submit=st.button('Predict')
if submit:
    start=time.time()
    prediction=model.predict([tweet])
    end=time.time()
    st.write('Prediction time taken: ',round(end-start,2),'seconds')
    print(prediction[0])
    st.write('Prediction sentiment is ',prediction[0])
