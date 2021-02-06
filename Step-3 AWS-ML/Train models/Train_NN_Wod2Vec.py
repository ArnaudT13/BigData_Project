'''
Description:
    Performs NLP on CV/resume description recognition and job prediction by using Word2Vec, training a Neural Network.

    Warning !! The Word2Vec model must be downloaded here : https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz and place in the same script location
'''

# for data
import json
import pandas as pd
import numpy as np

# for processing
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer

# for bag-of-words
from sklearn import feature_extraction, model_selection, preprocessing, feature_selection

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from joblib import dump, load

# word2vec
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim import matutils

## Data acquisition and preprocessing

'''
Preprocess a string.
:parameter
    :param text: string - name of column containing text
    :param lst_stopwords: list - list of stopwords to remove
    :param flg_stemm: bool - whether stemming is to be applied
    :param flg_lemm: bool - whether lemmitisation is to be applied
:return
    cleaned text
'''
def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, number=False, lst_stopwords=None):
    # clean (remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).strip()) # keep the upper case

    # remove number
    if not number:
        text = re.sub('[0-9]+', '', text)

    # Tokenize (convert from string to list)
    lst_text = text.split()

    # remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]

    # Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.SnowballStemmer("english")
        lst_text = [ps.stem(word) for word in lst_text]

    # Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    # back to string from list
    text = " ".join(lst_text)
    return text


# Loading data, labels and categories
print('[INFO] Loading data, labels and categories')
data     = pd.read_json("../Notebook/data.json")
category = pd.read_csv("../Notebook/categories_string.csv")
label    = pd.read_csv("../Notebook/label.csv")


# Merging data
data = pd.merge(data, label, how="right", on="Id")

# Downloading stopword from nltk
print('[INFO] Downloading stopword from nltk')
nltk.download('stopwords')
nltk.download('wordnet')

lst_stopwords = nltk.corpus.stopwords.words("english")

# Preprocessing data (cleaning data)
print('[INFO] Preprocessing data (cleaning data)')
data_clean = data
data_clean["description_clean"] = data_clean["description"].apply(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, number=True, lst_stopwords=lst_stopwords))

## Word2Vec

# Retrieve Word2Vec model
print('[INFO] Importing Word2Vec Model')
wv = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
wv.init_sims(replace=True)

# Downloading punkt from nltk
nltk.download('punkt')

# Function averaging for a CV extract
def word_averaging(wv, words):
    all_words, mean = set(), []

    # For each word in the cv extract
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        return np.zeros(wv.vector_size,)

    mean = matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

# Function averaging apply to all CV extracts
def word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, cv_extract) for cv_extract in text_list ])


# Tokenization CV extracts
def w2v_tokenize_text(text):
    tokens = []
    # For each sentence in the text
    for sent in nltk.sent_tokenize(text, language='english'):
        # For each words in the text
        for word in nltk.word_tokenize(sent, language='english'):
            # Delete words composed of 2 letters
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens


# Splitting dataset with the same seed
train, test = train_test_split(data_clean, test_size=0.2, random_state=211101)

# Tokenization
print('[INFO] Tokenization')
test_tokenized = test.apply(lambda r: w2v_tokenize_text(r['description_clean']), axis=1).values
train_tokenized = train.apply(lambda r: w2v_tokenize_text(r['description_clean']), axis=1).values

# Averaging
print('[INFO] Averaging')
X_train_word_average = word_averaging_list(wv, train_tokenized)
X_test_word_average = word_averaging_list(wv, test_tokenized)

## Classification

# Classification with a Neural Network : 1 layer of 50 neurons, RELU, strong penalization (alpha = 3)
print('[INFO] Classification (this may take time)')
nn_classifier = MLPClassifier(solver="lbfgs", hidden_layer_sizes=(30), alpha=1, activation="relu", max_iter=5000, random_state=0)
nn_classifier.fit(X_train_word_average, train['Category'])

# Export classifier
print('[INFO] Export classifier')
dump(nn_classifier, 'Models/NN_Word2Vec.joblib')