import re
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

nltk.download("wordnet")
nltk.download("omw-1.4")
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def remove_punctuations(text):
    return re.sub("[%s]" % re.escape(string.punctuation), "", text)


def remove_digits(text):
    return re.sub("W*dw*", "", text)


def remove_extra_spaces(text):
    return re.sub(" +", " ", text)


def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])


def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


# Reference: https://www.analyticsvidhya.com/blog/2021/06/must-known-techniques-for-text-preprocessing-in-nlp/
def preprocess_texts(texts):
    """
    texts (list): list of texts
    """
    for idx in tqdm(range(len(texts))):
        texts[idx] = texts[idx].lower()
        texts[idx] = remove_punctuations(texts[idx])
        texts[idx] = remove_digits(texts[idx])
        texts[idx] = remove_extra_spaces(texts[idx])
        texts[idx] = stem_words(texts[idx])
        texts[idx] = lemmatize_words(texts[idx])
    return texts
