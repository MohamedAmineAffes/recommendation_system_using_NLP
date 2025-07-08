import pandas as pd
import numpy as np
import nltk
import string
import re
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords")
nltk.download("wordnet")

# Preprocessing Functions

def first_sentence(text):
    lst = str(text).split(".")
    return lst[0] if lst else ""

def supprimer_numbers(text):
    return re.sub(r"[0-9]+", "", str(text))

def remove_punctuation(text):
    no_punct = [char for char in text if char not in string.punctuation]
    return "".join(no_punct)

def remove_stopwords(text):
    words = text.split()
    return [word for word in words if word not in stopwords]

def tokenize(text):
    return re.split(r"\W+", text)

def lemmatization(lst):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in lst]

def remove_special_words(lst):
    special_words = [
        "tv", "fire", "kindle", "last", "year", "amazon", "prime",
        "ipad", "app", "apple", "headphone", "netflix", "paperwhite",
        "echo", "alexa", "time", "phone", "tap", "one", "tangle",
        "speaker", "apps", "theyre", "case", "going", "ear", "portable",
        "bluetooth", "new", "back", "voice", "search", "cover", "got",
        "wifi", "book", "reading", "read", "capacity", "light",
        "controller", "screen", "web", "movie", "voyage", "surfing",
        "roku", "device", "samsung", "gaming", "purchased"
    ]
    return [word for word in lst if word not in special_words]

# Setup Stopwords
user_defined_stop_words = ["product", "Amazon"]
stopwords = list(set(nltk.corpus.stopwords.words("english")).union(string.punctuation, user_defined_stop_words))

# Load and preprocess data
df = pd.read_csv("7817_1.csv", engine="python", encoding="utf-8")
df["first_sentence"] = df["reviews.text"].apply(first_sentence)
df["no_number"] = df["first_sentence"].apply(supprimer_numbers)
df1 = df[["no_number", "reviews.title"]].drop_duplicates()

df1["text_lowercase"] = df1["no_number"].str.lower()
df1["no_punctuation"] = df1["text_lowercase"].apply(remove_punctuation)
df1["no_stopwords"] = df1["no_punctuation"].apply(remove_stopwords)
df1["no_extraspaces"] = df1["no_stopwords"].apply(lambda x: " ".join(x))
df1["tokenized"] = df1["no_extraspaces"].apply(tokenize)
df1["lemmatized"] = df1["tokenized"].apply(lemmatization)
df1["no_special_words"] = df1["lemmatized"].apply(remove_special_words)
df1["sentence"] = df1["no_special_words"].apply(lambda x: " ".join(x))

corpus = df1["sentence"].tolist()
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
df2 = df1.drop_duplicates(subset=["sentence"])
indices = pd.Series(df2.index, index=df2["sentence"]).drop_duplicates()


def get_recommendations(text, top_n=5):
    # Preprocess the input text similarly to the dataset
    text = text.lower()
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = " ".join(text)
    text = tokenize(text)
    text = lemmatization(text)
    text = remove_special_words(text)
    text = " ".join(text)
    # Transform the input using TF-IDF and compute similarity
    input_vec = vectorizer.transform([text])
    sim_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()
    sim_indices = sim_scores.argsort()[::-1][1 : top_n + 1]

    return df1["sentence"].iloc[sim_indices].tolist()
