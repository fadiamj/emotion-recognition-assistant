import openai
import numpy as np
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_explanation(predicted_label, predictions, features, model_accuracy=80):
    class_names = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad"}
    
    
    prompt = f"""
    A speech emotion recognition system predicted the emotion '{predicted_label}' for an audio sample. 
    The system provides probabilities for four possible emotions: Angry, Happy, Neutral, and Sad. 
    The system has an accuracy of {model_accuracy}%.
    
    The task is to explain why the system predicted the emotion '{predicted_label}', based on features such as tone, pitch, loudness, and other characteristics of the audio.

    Do not try to validate or change the predicted emotion. Just explain why this emotion was chosen, focusing on the characteristics of the audio sample.

    Output Format:
    - Emotion: (Just repeat the predicted emotion '{predicted_label}')
    - Explanation: (Provide a brief and simple explanation for why the system predicted the emotion based on features like tone, pitch, loudness, etc.)

    Only provide the emotion and explanation, without referring to probabilities or technical details.
    """

    messages = [
        {"role": "system", "content": "You are an assistant that helps with emotion recognition from speech."},
        {"role": "user", "content": prompt}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  
        messages=messages, 
        max_tokens=150,
        temperature=0.7
    )

    llm_output = response['choices'][0]['message']['content'].strip()

    emotion, explanation = llm_output.split("\n", 1) if "\n" in llm_output else (llm_output, "")

    emotion = emotion.replace("Emotion: ", "").strip()
    explanation = explanation.replace("Explanation: ", "").strip()

    return emotion, explanation



def explain_sentiment(sentiment_prediction, input_text, model_accuracy=94):
    prompt = f"""
    A sentiment analysis system predicted the sentiment '{sentiment_prediction}' for the following text: 
    "{input_text}". The system has an accuracy of {model_accuracy}%.

    Based on the text and the prediction, explain why this sentiment was chosen. If the sentiment is positive, 
    describe the emotion behind it (e.g., joy, excitement) and mention specific words or phrases in the text 
    that support this choice. Similarly, if the sentiment is negative, describe the associated emotion 
    (e.g., anger, sadness) and highlight supporting words. For neutral, explain why the text doesn't express 
    strong emotions.

    Output Format:
    - Sentiment: (Positive, Negative, or Neutral)
    - Explanation: (Provide a brief explanation in simple language, referencing the text when possible)
    
    Only provide the sentiment and explanation in the specified format.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.7
    )
    
    llm_output = response['choices'][0]['message']['content'].strip()
    
    sentiment, explanation = llm_output.split("\n", 1) if "\n" in llm_output else (llm_output, "")
    
    sentiment = sentiment.replace("Sentiment: ", "").strip()
    explanation = explanation.replace("Explanation: ", "").strip()

    return sentiment, explanation

def correct_grammar_and_spelling(text):
    try:
        # Call OpenAI GPT-3/4 for grammar and spelling correction
        response = openai.Completion.create(
            engine="text-davinci-003",  # You can replace with GPT-4 if available
            prompt=f"Correct the following text for grammar, spelling, and clarity: {text}",
            max_tokens=200,
            temperature=0.7
        )
        
        # Extract the corrected text from the response
        corrected_text = response.choices[0].text.strip()
        return corrected_text

    except Exception as e:
        print(f"Error with LLM grammar correction: {e}")
        return text 