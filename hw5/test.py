import re  # Regex
import string  # String manipulation
import spacy  # NLP tooling
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf  # Deep learning
import matplotlib.pyplot as plt  # Visualization
from sklearn.model_selection import train_test_split  # Split train and validation set
from sklearn.preprocessing import LabelEncoder  # Encode label into numerical format

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk("./hw5/dataset"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# nlp = spacy.load('en_core_web_sm')
df_train = pd.read_csv("./hw5/dataset/train.csv")
df_test = pd.read_csv("./hw5/dataset/test.csv")


def clean_text(text):
    regex_html = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    remove_digits = str.maketrans("", "", string.digits + string.punctuation)
    text = re.sub(regex_html, "", text)
    text = text.translate(remove_digits)
    return " ".join(
        re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split()
    ).lower()


df_train["text"] = (df_train.Title + " " + df_train.Description).apply(clean_text)
# \.apply(nlp).apply(lambda x: " ".join(token.lemma_ for token in x if not token.is_stop).lower())
df_test["text"] = (df_test.Title + " " + df_test.Description).apply(clean_text)
# \.apply(nlp).apply(lambda x: " ".join(token.lemma_ for token in x if not token.is_stop).lower())

le = LabelEncoder().fit(df_train["Class Index"])
df_train["label"] = le.transform(df_train["Class Index"])
df_test["label"] = le.transform(df_test["Class Index"])
train, val = train_test_split(
    df_train, test_size=0.2, random_state=42, stratify=df_train.label
)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=8192, oov_token="-")
tokenizer.fit_on_texts(train.text)

x_train = tf.keras.preprocessing.sequence.pad_sequences(
    tokenizer.texts_to_sequences(train.text),
    maxlen=128,
    padding="post",
    truncating="post",
)
x_val = tf.keras.preprocessing.sequence.pad_sequences(
    tokenizer.texts_to_sequences(val.text),
    maxlen=128,
    padding="post",
    truncating="post",
)
x_test = tf.keras.preprocessing.sequence.pad_sequences(
    tokenizer.texts_to_sequences(df_test.text),
    maxlen=128,
    padding="post",
    truncating="post",
)

y_train = train.label
y_val = val.label
y_test = df_test.label

from pdb import set_trace as bp

bp()

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input((128,)),
        tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index), output_dim=64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.GlobalMaxPool1D(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(set(df_train.label)), activation="softmax"),
    ]
)

model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

model.summary()

history = model.fit(
    x_train,
    y_train,
    epochs=50,
    validation_data=(x_val, y_val),
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2),
    ],
)

bp()
