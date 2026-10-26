import librosa
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer

with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

nltk.download('stopwords')

def preprocess_text(tweet, max_len=50):
    tweet = tweet.lower()
    tweet = re.sub(r"n't", " NEG", tweet)
    tweet = re.sub(r"\bnot\b", " NEG", tweet)
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'(.)\1{2,}', r'\1\1', tweet)  
    tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)
    
    stop_words = set(stopwords.words('english'))
    words = tweet.split()
    words = [word for word in words if word not in stop_words]
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]
    tweet = ' '.join(words)
    
    tweet_seq = tokenizer.texts_to_sequences([tweet])
    tweet_pad = pad_sequences(tweet_seq, maxlen=max_len, padding='post')

    return tweet_pad[0]


def preprocess_audio(file_path, sr=22050, target_duration=3, target_size=(256, 256)):
    y, _ = librosa.load(file_path, sr=sr)
  
    intervals = librosa.effects.split(y, top_db=20)
    y_trimmed = np.concatenate([y[start:end] for start, end in intervals])

    target_length = sr * target_duration
    if len(y_trimmed) < target_length:
        y_trimmed = np.pad(y_trimmed, (0, target_length - len(y_trimmed)))  
    elif len(y_trimmed) > target_length:
        y_trimmed = y_trimmed[:target_length]  

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)
    log_mel_spec = librosa.power_to_db(np.abs(mel_spec), ref=np.max)

    log_mel_spec_tensor = tf.convert_to_tensor(log_mel_spec, dtype=tf.float32)
    log_mel_spec_tensor = tf.expand_dims(log_mel_spec_tensor, axis=-1)  

    resized_spec = tf.image.resize(log_mel_spec_tensor, target_size)

    min_val = tf.reduce_min(resized_spec)
    max_val = tf.reduce_max(resized_spec)
    normalized_spec = 2 * (resized_spec - min_val) / (max_val - min_val) - 1

    normalized_spec_3channels = tf.concat([normalized_spec] * 3, axis=-1) 

    # Add batch dimension
    normalized_spec_3channels = tf.expand_dims(normalized_spec_3channels, axis=0) 

    return normalized_spec_3channels.numpy()


def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    pitches = librosa.pyin(y, fmin=75, fmax=500)
    pitch = np.nanmean(pitches) if pitches is not None else 0

    features = {
        'mfccs': np.mean(mfccs, axis=1).tolist(),
        'chroma': np.mean(chroma, axis=1).tolist(),
        'mel': np.mean(mel, axis=1).tolist(),
        'spectral_centroid': np.mean(spectral_centroid),
        'spectral_rolloff': np.mean(spectral_rolloff),
        'zcr': np.mean(zcr),
        'pitch': pitch
    }

    return features