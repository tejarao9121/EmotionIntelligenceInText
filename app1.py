# import streamlit as st
# import pandas as pd
# import numpy as np
# import altair as alt
# import joblib

# # Load the pre-trained model
# pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

# # Dictionary for emotion emojis
# emotions_emoji_dict = {
#     "anger": "😠",
#     "disgust": "🤮",
#     "fear": "😨😱",
#     "happy": "🤗",
#     "joy": "😂",
#     "neutral": "😐",
#     "sad": "😔",
#     "sadness": "😔",
#     "shame": "😳",
#     "surprise": "😮"
# }

# def predict_emotions(docx):
#     results = pipe_lr.predict([docx])
#     return results[0]

# def get_prediction_proba(docx):
#     results = pipe_lr.predict_proba([docx])
#     return results

# def main():
#     st.title("Text Emotion Detection")
#     st.subheader("Detect Emotions In Text")

#     with st.form(key='my_form'):
#         raw_text = st.text_area("Type Here")
#         submit_text = st.form_submit_button(label='Submit')

#     if submit_text:
#         col1, col2 = st.columns(2)

#         # Get prediction and probabilities
#         prediction = predict_emotions(raw_text)
#         probability = get_prediction_proba(raw_text)

#         with col1:
#             st.success("Original Text")
#             st.write(raw_text)

#             st.success("Prediction")
#             emoji_icon = emotions_emoji_dict.get(prediction, "😶")  # Default emoji if not found
#             st.write("{}: {}".format(prediction, emoji_icon))
#             st.write("Confidence: {:.2f}".format(np.max(probability)))

#         with col2:
#             st.success("Prediction Probability")

#             # Create a DataFrame from the prediction probabilities
#             proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
#             proba_df_clean = proba_df.T.reset_index()
#             proba_df_clean.columns = ["emotions", "probability"]

#             # Debugging information
#             st.write("Prediction Probability DataFrame:", proba_df_clean)
#             st.write("DataFrame Shape:", proba_df_clean.shape)
#             st.write("Data Types:\n", proba_df_clean.dtypes)

#             # Ensure correct data types and check for NaNs
#             try:
#                 # Force types
#                 proba_df_clean['probability'] = proba_df_clean['probability'].astype(float)
#                 proba_df_clean['emotions'] = proba_df_clean['emotions'].astype(str)
                
#                 # Debugging: Check if there are any NaN values
#                 if proba_df_clean.isnull().values.any():
#                     st.error("The probability DataFrame contains NaN values.")
#                     st.write(proba_df_clean[proba_df_clean.isnull().any(axis=1)])
#                     return

#                 # Check contents of the DataFrame before plotting
#                 st.write("Final DataFrame for Chart:", proba_df_clean)

#                 # Create and display the Altair chart
#                 fig = alt.Chart(proba_df_clean).mark_bar().encode(
#                     x=alt.X('emotions:N', sort=None),  # Specify categorical type
#                     y=alt.Y('probability:Q', axis=alt.Axis(title='Probability')),  # Specify quantitative type
#                     color='emotions:N'
#                 ).properties(title='Prediction Probability Distribution')

#                 st.altair_chart(fig, use_container_width=True)

#             except Exception as e:
#                 st.error(f"An error occurred while rendering the chart: {e}")
#                 st.write("Debugging Info: Check the data being passed to the chart.")
#                 st.write(proba_df_clean)

# if __name__ == '__main__':
#     main()
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Load the pre-trained model
pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

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

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        # Get prediction and probabilities
        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict.get(prediction, "😶")  # Default emoji if not found
            st.write("{}: {}".format(prediction, emoji_icon))
            st.write("Confidence: {:.2f}".format(np.max(probability)))

        with col2:
            st.success("Prediction Probability")

            # Create a DataFrame from the prediction probabilities
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            # Debug: Check the contents of proba_df_clean
            st.write("Prediction Probability DataFrame:", proba_df_clean)

            # Ensure correct data types and check for NaNs
            proba_df_clean['probability'] = proba_df_clean['probability'].astype(float)
            proba_df_clean['emotions'] = proba_df_clean['emotions'].astype(str)

            # Check for NaN values in the DataFrame
            if proba_df_clean.isnull().values.any():
                st.error("The probability DataFrame contains NaN values.")
                st.write(proba_df_clean[proba_df_clean.isnull().any(axis=1)])
                return

            # Create and display the Altair chart
            try:
                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x=alt.X('emotions:N', sort=None),  # Specify categorical type
                    y=alt.Y('probability:Q', axis=alt.Axis(title='Probability')),  # Specify quantitative type
                    color='emotions:N'
                ).properties(title='Prediction Probability Distribution')
                
                st.altair_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred while rendering the chart: {e}")

if __name__ == '__main__':
    main()
