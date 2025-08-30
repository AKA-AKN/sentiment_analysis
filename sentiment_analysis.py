import re
import warnings
import itertools
import numpy as np 
import pandas as pd 
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import gc
import pathlib

warnings.filterwarnings("ignore", category=FutureWarning)


# Load dataset
print("Loading dataset...")
data = pd.read_csv("Tweets.csv")
df = data[["text","airline_sentiment"]]
df.loc[:, 'text'] = df['text'].map(lambda x: x.lstrip('@VirginAmerica@UnitedAir@Southwestairline@DeltaAir@USAirways@American').rstrip('@'))
print("Dataset loaded successfully!")

# Remove neutral responses
df = df[df.airline_sentiment != "neutral"]
df['text'] = df['text'].apply(lambda x: x.lower()) # Convert to lowercase
df['text'] = df['text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x)) # Keep only alphanumeric chars

# Drop excessive negative samples
df = df.drop(df[df.airline_sentiment == "negative"].iloc[:5000].index)

print("Preprocessing completed!")

# Tokenization
max_features = 2000
print("Starting tokenization...")
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(df['text'].values)
X = tokenizer.texts_to_sequences(df['text'].values)
X = pad_sequences(X)
Y = df['airline_sentiment'].values

# Force garbage collection
gc.collect()
print("Tokenization completed!")

# Convert labels to binary
label_map = {"negative": 0, "positive": 1}
Y = np.array([label_map[label] for label in Y])

# Train-test split
print("Splitting dataset...")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=1)
print("Dataset split completed!")

# Define model
embed_dim = 128
lstm_out = 196
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(max_features, embed_dim, input_length=X_train.shape[1]))
model.add(tf.keras.layers.SpatialDropout1D(0.5))
model.add(tf.keras.layers.LSTM(lstm_out, dropout=0.3, recurrent_dropout=0.3))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(2, activation='softmax'))

# Compile model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train model
print("Starting model training...")
history = model.fit(X_train, Y_train, epochs=20, batch_size=16, validation_split=0.2, verbose=2)
print("Model training completed!")

# Save model & tokenizer
model.save("sentiment_model.h5")
np.save("tokenizer.npy", tokenizer.word_index)
print("Model and tokenizer saved!")

# Save training curves
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("results/accuracy_curve.png")
plt.clf()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("results/loss_curve.png")
plt.clf()

# Evaluate model
score = model.evaluate(X_test, Y_test, verbose=False)
print("Loss =", score[0])
print("Accuracy =", score[1])

with open("results/metrics.txt", "w") as f:
    f.write(f"Test Loss: {score[0]:.4f}\n")
    f.write(f"Test Accuracy: {score[1]:.4f}\n")

# Confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
confusion_mtx = confusion_matrix(Y_test, y_pred_classes)
plot_confusion_matrix(confusion_mtx, classes=["negative", "positive"])
plt.savefig("results/confusion_matrix.png")
plt.clf()

# Sample predictions
sample_texts = df['text'].iloc[:20].tolist()
sample_preds = []
for t in sample_texts:
    seq = tokenizer.texts_to_sequences([t])
    padded = pad_sequences(seq, maxlen=X_train.shape[1], dtype='int32', value=0)
    pred = model.predict(padded, verbose=0)[0]
    sample_preds.append("negative" if np.argmax(pred) == 0 else "positive")

pd.DataFrame({"Text": sample_texts, "Prediction": sample_preds}).to_csv("results/sample_predictions.csv", index=False)

print("All results saved in 'results/' folder.")
