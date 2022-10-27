import numpy as np
import tensorflow as tf
import pickle
import os

from datasets import load_dataset

from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import preprocess_texts


def get_dataset():
    if not os.path.exists("./hw5/dataset.pickle"):
        print("Encoded dataset not found. Encoding...")

        # Load dataset
        dataset = load_dataset("ag_news")

        # Tokenize words
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=8192, oov_token="-")

        X_train = preprocess_texts(dataset["train"]["text"])
        tokenizer.fit_on_texts(X_train)
        X_train = pad_sequences(
            tokenizer.texts_to_sequences(X_train),
            maxlen=128,
            padding="post",
            truncating="post",
        )
        X_test = preprocess_texts(dataset["test"]["text"])
        X_test = pad_sequences(
            tokenizer.texts_to_sequences(X_test),
            maxlen=128,
            padding="post",
            truncating="post",
        )
        y_train = dataset["train"]["label"]
        y_test = dataset["test"]["label"]
        dataset = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }
        with open("hw5/dataset.pickle", "wb") as f:
            pickle.dump(dataset, f)
        with open("hw5/tokenizer.pickle", "wb") as f:
            pickle.dump(tokenizer, f)

    else:
        print("Encoded dataset found")
        with open("hw5/dataset.pickle", "rb") as f:
            dataset = pickle.load(f)
        with open("hw5/tokenizer.pickle", "rb") as f:
            tokenizer = pickle.load(f)

    return dataset, tokenizer
