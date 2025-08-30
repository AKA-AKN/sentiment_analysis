import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Load the model and tokenizer
def load_model():
    if not os.path.exists("sentiment_model.h5") or not os.path.exists("tokenizer.npy"):
        print("Model or tokenizer file not found! Train the model first.")
        exit()
    
    model = tf.keras.models.load_model("sentiment_model.h5")
    tokenizer = Tokenizer()
    tokenizer.word_index = np.load("tokenizer.npy", allow_pickle=True).item()

    # Get maxlen safely from the model's input shape
    try:
        maxlen = model.input_shape[1]
    except Exception:
        maxlen = None
    
    if maxlen is None:
        raise ValueError("Could not infer maxlen from model. Please set it manually.")

    return model, tokenizer, maxlen



# Predict sentiment
def predict_sentiment(model, tokenizer, text, maxlen):
    text = text.lower()
    text_seq = tokenizer.texts_to_sequences([text])
    text_padded = pad_sequences(text_seq, maxlen=maxlen, dtype='int32', value=0)
    sentiment = model.predict(text_padded, batch_size=1, verbose=0)[0]
    return "negative" if np.argmax(sentiment) == 0 else "positive"

if __name__ == "__main__":
    model, tokenizer, maxlen = load_model()
    
    # Interactive loop
    while True:
        user_input = input("Enter text for sentiment analysis (or 'exit' to stop): ").strip()
        if user_input.lower() == "exit":
            break
        prediction = predict_sentiment(model, tokenizer, user_input, maxlen)
        print(f"Predicted Sentiment: {prediction}")
