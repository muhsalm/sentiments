import streamlit as st
import pandas as pd
import joblib
from utils import preprocessor

def run():
    # Load the saved model
    model = joblib.load('model.joblib')

    st.title("Sentiment Analysis")
    st.text("Basic app to detect the sentiment of text.")
    st.text("")
    userinput = st.text_input('Enter text below, then click the Predict button.', placeholder='Input text HERE')
    st.text("")
    predicted_sentiment = ""
    if st.button("Predict"):
        # Preprocess the user input
        input_series = pd.Series([userinput])
        preprocessed_input = preprocessor().transform(input_series)
        
        # Make prediction
        predicted_sentiment = model.predict(preprocessed_input)[0]
        
        if predicted_sentiment == 1:
            output = 'positive 👍'
        else:
            output = 'negative 👎'
        sentiment = f'Predicted sentiment of "{userinput}" is {output}.'
        st.success(sentiment)

if __name__ == "__main__":
    run()
