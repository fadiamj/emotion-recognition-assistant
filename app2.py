import streamlit as st
import tensorflow as tf
import numpy as np
import speech_recognition as sr
from preprocess import preprocess_audio, preprocess_text, extract_features
from customized_loss import penalized_loss
from LLM import generate_explanation, explain_sentiment, correct_grammar_and_spelling
from attention import Attention, se_block

# Load the pre-trained models
@st.cache_resource
def load_model1():
    model = tf.keras.models.load_model('pen2_model.h5', custom_objects={'penalized_loss': penalized_loss, 'se_block': se_block})
    return model

model1 = load_model1()

@st.cache_resource

def load_model2():
    model = tf.keras.models.load_model('sentiment_analysis2.h5', custom_objects={'Attention':Attention})
    return model

model2 = load_model2()

# Emotion and Sentiment Class Mappings
speech_class = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad"}
text_class = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Navigation Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Main Page", "Audio Analysis", "Text Analysis"])

# Main Page
if page == "Main Page":
    st.title("Welcome to the Emotion Recognition Assistant")
    st.write("This app allows you to analyze emotions from audio and sentiment from text.")
    st.write("Navigate to the respective pages using the sidebar to begin your analysis.")

# Audio Analysis Page
elif page == "Audio Analysis":
    st.title("Audio Emotion Recognition & Speech-to-Text")
    st.write("Upload an audio file to analyze the emotion in it or convert speech to text.")

    uploaded_audio = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

    if uploaded_audio is not None:
        st.audio(uploaded_audio, format='audio/wav')  
        temp_file = f"temp_{uploaded_audio.name}"
        with open(temp_file, "wb") as f:
            f.write(uploaded_audio.read())
        
        # Emotion from Tone Analysis
        st.subheader("Analyze Emotion from Tone")
        if st.button("Analyze Emotion from Tone"):
            features = extract_features(temp_file)
            preprocessed_audio = preprocess_audio(temp_file)
            
            predictions = model1.predict(preprocessed_audio)
            predicted_class = np.argmax(predictions, axis=-1)
            predicted_label = speech_class.get(predicted_class[0], "Unknown")
            
            refined_emotion, explanation = generate_explanation(
                predicted_label=predicted_label,
                predictions=predictions.tolist()[0], 
                features=features
            )
            
            st.write(f"**Emotion Detected (Tone):** {refined_emotion}")
            st.write(f"**Explanation:** {explanation}")
        
        # Speech-to-Text Conversion & Emotion from Context
        st.subheader("Analyze Emotion from Context")
        if st.button("Convert Speech to Text and Analyze Emotion"):
            # Using SpeechRecognition to convert audio to text
            recognizer = sr.Recognizer()

            with sr.AudioFile(temp_file) as source:
                audio_data = recognizer.record(source)
            
            try:
                # Convert speech to text
                transcribed_text = recognizer.recognize_google(audio_data)

                # Correct the grammar and spelling using LLM
                corrected_text = correct_grammar_and_spelling(transcribed_text)
                st.write("**Transcribed Text:**")
                st.write(corrected_text) 

                # Analyze sentiment (emotion) from the corrected text using the sentiment analysis model
                preprocessed_text = preprocess_text(corrected_text)
                predictions = model2.predict(np.array([preprocessed_text]))
                predicted_class = np.argmax(predictions, axis=-1)
                predicted_label = text_class.get(predicted_class[0], "Unknown")
                
                refined_sentiment, explanation = explain_sentiment(predicted_label, corrected_text)
                
                st.write(f"**Emotion Detected (Context):** {refined_sentiment}")
                st.write(f"**Explanation:** {explanation}")
            
            except sr.UnknownValueError:
                st.write("Google Speech Recognition could not understand the audio")
            except sr.RequestError as e:
                st.write(f"Could not request results from Google Speech Recognition service; {e}")

# Text Analysis Page
elif page == "Text Analysis":
    st.title("Text Sentiment Analysis")
    st.write("Enter a text snippet to analyze its sentiment.")

    user_text = st.text_area("Enter text for sentiment analysis")

    if user_text:
        if st.button("Analyze Text"):
            preprocessed_text = preprocess_text(user_text)
            predictions = model2.predict(np.array([preprocessed_text]))
            predicted_class = np.argmax(predictions, axis=-1)
            predicted_label = text_class.get(predicted_class[0], "Unknown")
            
            refined_sentiment, explanation = explain_sentiment(predicted_label, user_text)
            
            st.write(f"**Sentiment Detected:** {refined_sentiment}")
            st.write(f"**Explanation:** {explanation}")
