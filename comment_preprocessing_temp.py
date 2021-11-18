import numpy as np
import pandas as pd
import re
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import time


## load in dataset
df = pd.read_csv('comment_temp.csv')
df = df.rename({'body':'text'}, axis = 1)

stopwords_list = set(stopwords.words('english') + list(punctuation))


def clean_tweet(tweet, stopwords_list):
    porter = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    # tweet = tweet.encode("ascii", errors="ignore").decode()  # remove non-English words
    tweet = tweet.lower()  # convert text to lower-case
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', tweet)  # remove URLs
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)  # remove the # in #hashtag
    tweet = re.sub(r'<.*?>', '', tweet)  # remove HTML
    tweet = re.sub(r'(?:@[\w_]+)', '', tweet)  # remove @-mentions
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)', '', tweet)  # remove numbers
    tweet = re.sub(r'http\S+', '', tweet) # remove url

    word_list = word_tokenize(tweet)  ## tokenization
    word_list = [word for word in word_list if word not in stopwords_list]  # remove stopwords
    word_list = [porter.stem(word) for word in word_list]  # stemming
    word_list = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in word_list]  # lematizer
    word_list = [word for word in word_list if len(word) > 1]  # remove single-char word
    word_list = [word for word in word_list if len(word) < 15]  # remove long words
    word_list = " ".join(word_list)
    return word_list

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)



for i in range(len(df["text"])):
    df.loc[i, "text"] = clean_tweet(df.loc[i, "text"],stopwords_list)


def train_dev_split(df):
    df = df.dropna()
    # split the dataset in train and test
    X= df['text']
    #y = df['label']
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    return X_train, X_dev, y_train, y_dev

def vectorization(X_train, X_dev, X_test):
    # make Xs as vectors
    vectorizer = CountVectorizer(lowercase=False)
    vectorizer.fit(X_train)
    X_train_dtm = vectorizer.transform(X_train)
    X_dev_dtm = vectorizer.transform(X_dev)
    X_test_dtm = vectorizer.transform(X_test)
    return X_train_dtm,X_dev_dtm,X_test_dtm

vectorizer = CountVectorizer(lowercase=False)
vectorizer.fit(df['text'])

X1 = vectorizer.transform(df['text'])


# ===== then we can start with modeling ====== #

