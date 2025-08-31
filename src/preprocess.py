import pandas as pd
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from neattext.functions import remove_stopwords, remove_special_characters
from tqdm import tqdm

### ---------- Baseline Preprocessing (TF-IDF) ---------- ###
def clean_for_ml(text):
    """Light cleaning for classical ML (LogReg, SVM)."""
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    return text.lower().strip()

def prepare_baseline_data(path, test_size=0.2):
    df = pd.read_csv(path)
    df['clean_text'] = df['text'].apply(clean_for_ml)
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['class'],
        test_size=test_size,
        stratify=df['class'],
        random_state=42
    )
    return X_train, X_test, y_train, y_test

### ---------- Deep Learning Preprocessing (Sequences) ---------- ###
def clean_for_dl(texts):
    """Aggressive cleaning for deep learning models (RNN, CNN, Transformers)."""
    cleaned_text = []
    text_length = []
    for sent in tqdm(texts, desc="Cleaning DL texts"):
        sent = sent.lower()
        sent = remove_special_characters(sent)
        sent = remove_stopwords(sent)
        cleaned_text.append(sent)
        text_length.append(len(sent.split()))
    return cleaned_text, text_length

def prepare_deep_data(path, maxlen=50, test_size=0.2):
    df = pd.read_csv(path)

    # Clean texts
    cleaned_text, text_length = clean_for_dl(df['text'])
    df['clean_text'] = cleaned_text
    df['text_length'] = text_length

    # Split data
    train_data, test_data = train_test_split(
        df, test_size=test_size, stratify=df['class'], random_state=42
    )

    # Tokenize & pad
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data['clean_text'])

    train_seq = tokenizer.texts_to_sequences(train_data['clean_text'])
    test_seq = tokenizer.texts_to_sequences(test_data['clean_text'])

    train_pad = pad_sequences(train_seq, maxlen=maxlen)
    test_pad = pad_sequences(test_seq, maxlen=maxlen)

    return train_pad, test_pad, train_data['class'], test_data['class'], tokenizer
