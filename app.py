import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Load GRU model and tokenizer
# -------------------------------
model = load_model("gru_model.h5")
tokenizer = joblib.load("tokenizer.pkl")

# Parameters
MAX_LEN = 50  # must match the length used during training

# -------------------------------
# Prediction function
# -------------------------------
def predict(text):
    # Preprocess text
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Model prediction
    pred_prob = float(model.predict(padded)[0][0])  # convert numpy.float32 -> float
    pred_class = 1 if pred_prob >= 0.5 else 0
    return pred_class, pred_prob

# -------------------------------
# Streamlit app
# -------------------------------
def main():
    st.set_page_config(page_title="Suicide Risk Detection (GRU)", layout="centered")
    st.title("Suicide Risk Detection (GRU Model)")
    st.write("Enter text below to predict suicide risk:")

    text = st.text_area("Enter text here:")

    if st.button("Predict"):
        if not text.strip():
            st.warning("Please enter some text!")
        else:
            pred, conf = predict(text)
            risk_label = "High Risk" if pred == 1 else "Low Risk"
            
            # Display results
            st.subheader(f"Prediction: {risk_label}")
            st.subheader(f"Confidence: {conf*100:.2f}%")
            
            # Colored progress bar (fixed float type)
            st.progress(min(conf, 1.0))

if __name__ == "__main__":
    main()