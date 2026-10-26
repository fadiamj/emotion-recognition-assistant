# Emotion Recognition Assistant 🎭

An AI-powered application that analyzes **emotions in speech** and **sentiment in text**, designed to help users better understand emotional cues in communication.

## 📌 Overview

Understanding emotions is an important part of effective communication. However, some individuals, including people with autism, alexithymia, or social anxiety, may find interpreting emotional cues challenging.

The Emotion Recognition Assistant combines deep learning, audio processing, natural language processing, and Large Language Models (LLMs) to identify emotional patterns and provide accessible explanations.

The application is built with **Python and Streamlit** and supports both audio-based emotion recognition and text-based sentiment analysis.

## ✨ Features

### 🎤 Speech Emotion Recognition
- Upload audio files for emotion analysis.
- Process audio and extract meaningful features using signal processing techniques.
- Convert audio signals into mel spectrograms for deep learning analysis.
- Use a Convolutional Neural Network (CNN) enhanced with Squeeze-and-Excitation (SE) blocks.
- Apply a custom loss function to address important emotion misclassifications.
- Achieved **80% test accuracy** for the final speech emotion recognition model during project evaluation.

### 📝 Text Sentiment Analysis
- Enter text for sentiment analysis.
- Classify text sentiment as:
  - Positive
  - Neutral
  - Negative
- Use a trained deep learning model to analyze textual input.

### 🤖 AI-Powered Explanations
- Integrate a Large Language Model (LLM) through the OpenAI API.
- Provide user-friendly explanations of detected emotions.
- Help users interpret emotional and sentiment predictions more easily.

### 🖥️ Interactive Interface
- Built with Streamlit.
- Provides a simple interface for submitting audio and text.
- Combines speech and text analysis in one application.

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Interactive web application |
| TensorFlow / Keras | Deep learning model development and inference |
| Convolutional Neural Networks (CNNs) | Speech emotion classification |
| Squeeze-and-Excitation (SE) Blocks | Emphasizing informative feature channels |
| Librosa | Audio processing and feature extraction |
| Mel Spectrograms | Audio representation for model input |
| Natural Language Processing (NLP) | Text sentiment analysis |
| OpenAI API | Language-model-powered explanations |
| Git & Git LFS | Version control and storage of large model files |

## 📂 Project Structure

```text
emotion-recognition-assistant/
│
├── app2.py                    # Main Streamlit application
├── LLM.py                     # Large Language Model integration
├── preprocess.py              # Text preprocessing utilities
├── attention.py               # Attention-related model components
├── customized_loss.py         # Custom loss function
│
├── pen2_model.h5              # Trained speech emotion model
├── sentiment_analysis2.h5     # Trained text sentiment model
├── tokenizer.pkl              # Saved text tokenizer
│
├── Final_model.ipynb          # Model development notebook
├── working_with_audio.ipynb   # Audio processing experiments
├── Report 499.docx            # Project report
│
├── .gitattributes             # Git LFS configuration
└── .gitignore                 # Files excluded from Git
```

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/fadiamj/emotion-recognition-assistant.git
cd emotion-recognition-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

Install the packages required by the application and its model components. Depending on the code and model versions, these may include:

```bash
pip install streamlit tensorflow librosa numpy scikit-learn openai
```

Additional packages may be required by the existing source code. The application should be tested with compatible dependency versions.

### 4. Configure the OpenAI API key

The application reads the API key from an environment variable rather than storing it directly in the source code.

On Windows Command Prompt:

```cmd
set OPENAI_API_KEY=your_api_key
```

On macOS or Linux:

```bash
export OPENAI_API_KEY="your_api_key"
```

Replace `your_api_key` with your own valid API key. Keep it private and never commit it to GitHub.

### 5. Run the application

```bash
streamlit run app2.py
```

Streamlit will provide a local URL where you can open the application in your browser.

## 📊 Model Development and Evaluation

The speech emotion recognition model was developed through several experiments, including:

- MFCC feature extraction.
- Audio augmentation using noise addition, pitch shifting, and time stretching.
- Mel spectrogram generation.
- Experiments with Vision Transformers (ViTs) and CNN architectures.
- Incorporation of additional speech datasets.
- Squeeze-and-Excitation blocks to improve feature representation.
- A custom loss function to reduce specific emotion misclassifications.
- Hyperparameter tuning and optimizer experimentation.

The final speech model achieved **80% test accuracy** in the reported evaluation.

The text sentiment component analyzes written input and predicts positive, neutral, or negative sentiment.

## 📚 Datasets

The speech emotion recognition experiments used the following datasets:

- CREMA-D
- SAVEE
- TESS
- RAVDESS
- EmoDB

These datasets provided speech samples with emotion labels for model training and evaluation.

## ⚠️ Limitations

- Emotion recognition accuracy depends on audio quality, speaker variation, and the diversity of the training data.
- The speech model experienced overfitting during development.
- Emotional expression can vary across cultures, languages, accents, and individuals.
- Text sentiment analysis may struggle with sarcasm, irony, humor, and context-dependent expressions.
- The application is an assistive tool, not a clinical diagnostic system or a definitive measure of a person's emotional state.

## 🚀 Future Improvements

Potential improvements include:

- Expanding the training datasets with more diverse speakers and emotional expressions.
- Reducing overfitting and improving model generalization.
- Supporting additional languages and dialects.
- Improving processing speed and real-time feedback.
- Enhancing contextual understanding of sarcasm, irony, and nuanced emotional expressions.
- Exploring additional transformer-based architectures and model optimization techniques.

## 🎯 Project Objective

The goal of this project is to explore how AI can support emotional awareness and communication by combining speech emotion recognition, text sentiment analysis, and accessible language-model explanations in a single application.

## 👩‍💻 Authors

**Fadia Alhajjeh & Rawan Ahmad**

Data Science & Artificial Intelligence Program  
Department of Information Systems  
Yarmouk University

**Academic Supervisor:** Dr. Enas Khashashneh

---

*Academic capstone project — First Term 2024.*

