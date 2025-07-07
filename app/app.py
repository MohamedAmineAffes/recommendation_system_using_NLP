import pandas as pd
import numpy as np
import nltk
import string
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from flask import Flask, request, jsonify, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

# Download required NLTK data
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.stem.wordnet import WordNetLemmatizer


# -------------------
# Your preprocessing functions
# -------------------

def first_sentence(text):
    lst = text.split(".")
    return lst[0]


def supprimer_numbers(text):
    return re.sub(r"[0-9]+", "", text)


def remove_stopwords(text):
    no_stopwords = [w for w in text.split() if w not in stopwords]
    return no_stopwords


def remove_punctuation(text):
    no_punct = [c for c in text if c not in string.punctuation]
    return "".join(no_punct)


def tokenize(text):
    return re.split("\W+", text)


def listToString(s):
    return " ".join(s)


def lemmatization(lst):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in lst]


def remove_special_words(lst1):
    lst2 = [
        "tv", "fire", "kindle", "last", "year", "amazon", "prime", "ipad",
        "app", "apple", "headphone", "netflix", "paperwhite", "echo", "alexa",
        "time", "phone", "tap", "one", "tangle", "speaker", "apps", "theyre",
        "case", "going", "ear", "portable", "bluetooth", "new", "back", "voice",
        "search", "cover", "got", "wifi", "book", "reading", "read", "capacity",
        "light", "controller", "screen", "web", "movie", "voyage", "surfing",
        "roku", "device", "samsung", "gaming", "purchased",
    ]
    return [word for word in lst1 if word not in lst2]


def get_recommendations(title, cosine_sim, indices, df1):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # top 10 excluding itself
    movie_indices = [i[0] for i in sim_scores]
    return df1["sentence"].iloc[movie_indices].tolist()


# -------------------
# Data loading and preprocessing (run once)
# -------------------

print("Loading and preprocessing data...")

df = pd.read_csv("7817_1.csv", engine="python", encoding="utf-8")

df["first_sentence"] = df["reviews.text"].apply(first_sentence)
df["no_number"] = df["first_sentence"].apply(supprimer_numbers)
df1 = df[["no_number", "reviews.title"]]
df1 = df1.drop_duplicates()

user_defined_stop_words = ["product", "Amazon"]
stopwords = set(nltk.corpus.stopwords.words("english")).union(
    set(string.punctuation + " ".join(user_defined_stop_words).lower()))

df1["text_lowercase"] = df1["no_number"].apply(lambda x: x.lower())
df1["no_punctuation"] = df1["text_lowercase"].apply(remove_punctuation)
df1["no_stopwords"] = df1["no_punctuation"].apply(remove_stopwords)
df1["no_extraspaces"] = df1["no_stopwords"].apply(listToString)
df1["tokenized"] = df1["no_extraspaces"].apply(tokenize)
df1["lemmatized"] = df1["tokenized"].apply(lemmatization)
df1["no_special_words"] = df1["lemmatized"].apply(remove_special_words)
df1["sentence"] = df1["no_special_words"].apply(listToString)

corpus = df1["sentence"].tolist()
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df1.index, index=df1["sentence"]).drop_duplicates()

print("Data loaded and preprocessed.")

# -------------------
# Flask App + Prometheus
# -------------------

app = Flask(__name__)

REQUEST_COUNT = Counter("flask_app_requests_total", "Total HTTP Requests", ["endpoint", "method", "http_status"])
REQUEST_LATENCY = Histogram("flask_app_request_latency_seconds", "Request latency", ["endpoint"])


@app.before_request
def start_timer():
    request.start_time = time.time()


@app.after_request
def record_metrics(response):
    resp_time = time.time() - request.start_time
    REQUEST_LATENCY.labels(request.path).observe(resp_time)
    REQUEST_COUNT.labels(request.path, request.method, response.status_code).inc()
    return response


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    if not data or "title" not in data:
        return jsonify({"error": "Missing 'title' in JSON body"}), 400

    title = data["title"]
    recs = get_recommendations(title, cosine_sim, indices, df1)
    if not recs:
        return jsonify({"error": "Title not found or no recommendations"}), 404
    return jsonify({"recommendations": recs})


@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
