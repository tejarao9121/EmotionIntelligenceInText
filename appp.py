import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import pickle
import time
import os

# Load the pre-trained models
# Make sure to use absolute paths or check if the file exists before loading
try:
    emotion_model_path = "model/text_emotion.pkl"
    # sentiment_model_path = "twitter_sentiment.pkl"

    if os.path.exists(emotion_model_path):
        emotion_model = joblib.load(open(emotion_model_path, "rb"))
    else:
        st.error("Emotion model file not found. Please check the path!")

    # if os.path.exists(sentiment_model_path):
    #     sentiment_model = pickle.load(open(sentiment_model_path, 'rb'))
    # else:
    #     st.error("Sentiment model file not found. Please check the path!")

except Exception as e:
    st.error(f"Error loading models: {e}")

# Dictionary for emotion emojis
emotions_emoji_dict = {
    "anger": "😠",
    "disgust": "🤮",
    "fear": "😨😱",
    "happy": "🤗",
    "joy": "😂",
    "neutral": "😐",
    "sad": "😔",
    "sadness": "😔",
    "shame": "😳",
    "surprise": "😮"
}

# Sentiment mapping based on emotions
# sentiment_mapping = {
#     "happy": "Positive",
#     "joy": "Positive",
#     "anger": "Negative",
#     "disgust": "Negative",
#     "fear": "Negative",
#     "sad": "Negative",
#     "sadness": "Negative",
#     "shame": "Negative",
#     "neutral": "Neutral",
#     "surprise": "Irrelevant"
# }

def predict_emotions(docx):
    try:
        results = emotion_model.predict([docx])
        return results[0]
    except Exception as e:
        st.error(f"Error in emotion prediction: {e}")

def get_prediction_proba(docx):
    try:
        results = emotion_model.predict_proba([docx])
        return results
    except Exception as e:
        st.error(f"Error in getting emotion probabilities: {e}")

def main():
    st.title("Emotion Intelligence In Text")
    st.subheader("Detect Emotions  in Text")

    # Input for text analysis
    raw_text = st.text_input("Type Here")

    if st.button('Predict'):
        col1, col2 = st.columns(2)

        # Emotion Analysis
        if raw_text:
            prediction_emotion = predict_emotions(raw_text)
            probability_emotion = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Emotion Prediction")
                emoji_icon = emotions_emoji_dict.get(prediction_emotion, "😶")  # Default emoji if not found
                st.write(f"{prediction_emotion}: {emoji_icon}")

                # Display confidence score
                if probability_emotion is not None:
                    st.write("Confidence: {:.2f}".format(np.max(probability_emotion)))

                    # Create a DataFrame for emotion prediction probabilities
                    proba_df_emotion = pd.DataFrame(probability_emotion, columns=emotion_model.classes_)
                    proba_df_clean_emotion = proba_df_emotion.T.reset_index()
                    proba_df_clean_emotion.columns = ["emotions", "probability"]

                    # Display chart for emotion prediction
                    try:
                        fig_emotion = alt.Chart(proba_df_clean_emotion).mark_bar().encode(
                            x=alt.X('emotions:N', sort=None),
                            y=alt.Y('probability:Q', axis=alt.Axis(title='Probability')),
                            color='emotions:N'
                        ).properties(title='Emotion Prediction Probability Distribution')

                        st.altair_chart(fig_emotion, use_container_width=True)
                    except Exception as e:
                        st.error(f"An error occurred while rendering the emotion chart: {e}")

        # Sentiment Analysis
        # if raw_text:
        #     start_time = time.time()
        #     try:
        #         prediction_sentiment = sentiment_model.predict([raw_text])
        #         end_time = time.time()

        #         with col2:
        #             st.success("Sentiment Prediction")
        #             sentiment = sentiment_mapping.get(prediction_emotion, "Unknown")
        #             st.write(f'Predicted sentiment is: {sentiment}')
        #             st.write('Prediction time taken: ', round(end_time - start_time, 2), 'seconds')

        #     except Exception as e:
        #         st.error(f"Error in sentiment prediction: {e}")

if __name__ == '__main__':
    main()
