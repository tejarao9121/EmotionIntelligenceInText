import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import pickle
import os

# Load pre-trained models
def load_model(file_path):
    try:
        return pickle.load(open(file_path, "rb"))
    except FileNotFoundError:
        st.error(f"Model file not found at {file_path}. Please check the path.")
        return None

# Construct paths dynamically
current_dir = os.path.dirname(__file__)
emotion_model_path = os.path.join(current_dir, "model", "text_emotion.pkl")
sentiment_model_path = os.path.join(current_dir, "model", "sentiment.pkl")

pipe_lr = joblib.load(emotion_model_path) if os.path.exists(emotion_model_path) else None
sentiment_model = load_model(sentiment_model_path)

# Dictionary for emotion emojis
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮"
}

def predict_emotions(docx):
    return pipe_lr.predict([docx])[0] if pipe_lr else None

def get_prediction_proba_emotion(docx):
    return pipe_lr.predict_proba([docx]) if pipe_lr else None

def predict_sentiment(docx):
    return sentiment_model.predict([docx])[0] if sentiment_model else None

def main():
    st.title("Text Emotion and Sentiment Detection")
    st.subheader("Analyze emotions and sentiments in text")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text and raw_text:
        col1, col2 = st.columns(2)

        # Get emotion prediction and probabilities
        emotion_prediction = predict_emotions(raw_text)
        emotion_probability = get_prediction_proba_emotion(raw_text)

        # Get sentiment prediction
        sentiment_prediction = predict_sentiment(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            if emotion_prediction:
                st.success("Emotion Prediction")
                emoji_icon = emotions_emoji_dict.get(emotion_prediction, "😶")
                st.write(f"Emotion: {emotion_prediction} {emoji_icon}")
                st.write("Emotion Confidence: {:.2f}".format(np.max(emotion_probability)))

            if sentiment_prediction:
                st.success("Sentiment Prediction")
                st.write(f"Sentiment: {sentiment_prediction}")

        if emotion_probability is not None:
            with col2:
                st.success("Emotion Prediction Probability")

                # Create a DataFrame from the prediction probabilities
                proba_df = pd.DataFrame(emotion_probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                # Plot emotion probability distribution
                try:
                    fig = alt.Chart(proba_df_clean).mark_bar().encode(
                        x=alt.X('emotions:N', sort=None),
                        y=alt.Y('probability:Q', axis=alt.Axis(title='Probability')),
                        color='emotions:N'
                    ).properties(title='Emotion Prediction Probability Distribution')
                    
                    st.altair_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"An error occurred while rendering the chart: {e}")

if __name__ == '__main__':
    main()
