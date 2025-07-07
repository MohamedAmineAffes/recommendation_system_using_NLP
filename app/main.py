import pandas as pd
import numpy as np
import nltk
import string
import re

# Install nltk
#!pip install -q wordcloud
# import wordcloud
from wordcloud import WordCloud
import nltk

nltk.download("stopwords")
nltk.download("wordnet")
from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plt

# % matplotlib inline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def first_sentence(text):
    lst = text.split(".")  # split text to a list according to "."
    return lst[0]


def supprimer_numbers(text):
    text = re.sub(r"[0-9]+", "", text)  # eliminate numbers from the text
    return text


# Removing stopwords
def remove_stopwords(text):
    no_stopwords = [
        words for words in text.split() if words not in stopwords
    ]  # create list of words without stopwords
    words_wo_stopwords = " ".join(no_stopwords)  #
    return no_stopwords  #


# Removing ponctuation
def remove_punctuation(text):
    no_punct = [
        words for words in text if words not in string.punctuation
    ]  # divise le texte en lettres en eliminant les ponctuations
    words_wo_punct = "".join(no_punct)  # concatine les mots
    return words_wo_punct


# Tokenize
def tokenize(text):
    split = re.split(
        "\W+", text
    )  # divise le texte en element suivant les espaces en les versant dans une liste
    return split


def listToString(s):
    str1 = " "  # initialize an empty string
    return str1.join(s)  # concatenate list to string


def lemmatization(lst):
    lst1 = []
    lemmatizer = WordNetLemmatizer()  # create instance for WordNetLemmatizer()
    for word in lst:  # run all the list
        word1 = lemmatizer.lemmatize(word)  # apply lemmatize()
        lst1.append(word1)  # add words to lst1
    return lst1  # lst1 contains


def remove_special_words(lst1):
    lst2 = [
        "tv",
        "fire",
        "kindle",
        "last",
        "year",
        "amazon",
        "prime",
        "ipad",
        "app",
        "apple",
        "headphone",
        "netflix",
        "paperwhite",
        "echo",
        "alexa",
        "time",
        "phone",
        "tap",
        "one",
        "tangle",
        "speaker",
        "apps",
        "theyre",
        "case",
        "going",
        "ear",
        "portable",
        "bluetooth",
        "new",
        "back",
        "voice",
        "search",
        "cover",
        "got",
        "wifi",
        "book",
        "reading",
        "read",
        "capacity",
        "light",
        "controller",
        "screen",
        "web",
        "movie",
        "voyage",
        "surfing",
        "roku",
        "device",
        "samsung",
        "gaming",
        "purchased",
    ]
    l = [
        word for word in lst1 if word not in lst2
    ]  # l doesn't contain any word from lst2
    return l


df = pd.read_csv(
    "7817_1.csv", engine="python", encoding="utf-8"
)  # read the datebase

df["first_sentence"] = df["reviews.text"].apply(
    first_sentence
)  # create new column containing first sentence from each comment


df["no_number"] = df["first_sentence"].apply(
    supprimer_numbers
)  # create new column containing sentences without numbers

df1 = df[
    ["no_number", "reviews.title"]
]  # create new dataframe containing both columns 'no_number' and 'title'


df1.drop_duplicates().shape  # from 2700 rows we get 1430 rows after deleting the duplicated rows

# Removing stopwords
user_defined_stop_words = ["product", "Amazon"]  # list for the user stopwords
list_of_amazon_product = [""]  # list for the amazon product

i = nltk.corpus.stopwords.words("english")  # list of english stopwords
j = (
    list(string.punctuation) + user_defined_stop_words
)  # cancatenated list of ponctuation and user stopwords

stopwords = set(i).union(j)  # create a set of i and j
stopwords = list(stopwords)  # transform to list


df1["text_lowercase"] = df1["no_number"].apply(
    lambda x: x.lower()
)  # create new column for lowercase text
df1[
    ["no_number", "text_lowercase"]
].head()  # print the 5 first lignes of 'no_number' and 'text_lowercase' columns
# removing ponctuation
df1["no_ponctuation"] = df1["text_lowercase"].apply(
    remove_punctuation
)  # create new column without ponctuation
df1[["text_lowercase", "no_ponctuation"]].head()
df1["no_stopwords"] = df1["no_ponctuation"].apply(
    remove_stopwords
)  # create new column without stopwords
df1[["no_ponctuation", "no_stopwords"]].head()


df1["no_extraspaces"] = df1["no_stopwords"].apply(
    listToString
)  # create new column containing sentences (string)
df1[["no_stopwords", "no_extraspaces"]].head()
# tokeniZation
df1["tokenized"] = df1["no_extraspaces"].apply(
    tokenize
)  # create new column of tokenzed sentences
df1[
    ["no_extraspaces", "tokenized"]
].head()  # print the first 5 lignes of 'no_extraspaces' and 'tokenized' columns

df1["lemmatized"] = df1["tokenized"].apply(
    lemmatization
)  # create new column with lemmatized list
df1[["tokenized", "lemmatized"]].head()


df1["no_special_words"] = df1["lemmatized"].apply(
    remove_special_words
)  # create new column without special words
df1["sentence"] = df1["no_special_words"].apply(
    listToString
)  # create new column containing sentence for each ligne
full_text = " "
full_text = full_text.join(df1["sentence"])  # concatenate all sentences
wordcloud = WordCloud(
    max_font_size=70, max_words=100, background_color="white"
).generate(
    full_text
)  # generates a wordcloud from the text
plt.figure(figsize=(10, 5))  # create a figure with (10,5) dimension
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")  # no axis
plt.title("WordCloud", size=50, color="red")  # make a title
plt.show()  # showing results
plt.savefig("WordCloud.png")
corpus = df1["sentence"].tolist()
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)
# print(vectorizer.get_feature_names())
# print(tfidf_matrix.shape)
#cols = vectorizer.get_feature_names()
cols = vectorizer.get_feature_names_out()

dense = tfidf_matrix.todense().tolist()
# print(dense)
embedded_df = pd.DataFrame(dense, columns=cols)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# print(cosine_sim)
df2 = df1.drop_duplicates(subset=["sentence"])
indices = pd.Series(df2.index, index=df2["sentence"]).drop_duplicates()


def get_recommendations(title, cosine_sim, indices):
    # Get the index of the movie that matches the title
    idx = indices[title]
    # Get the pairwsie similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    # print(sim_scores)
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores for 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return df1["sentence"].iloc[movie_indices]

print(indices.head(10))
get_recommendations(
    "initially trouble deciding review less said thing great spending money go",
    cosine_sim,
    indices,
)
