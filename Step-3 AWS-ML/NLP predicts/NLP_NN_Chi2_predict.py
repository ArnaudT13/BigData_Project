'''
Description:
    Performs NLP on CV/resume description recognition and job prediction
    by using TF-IDF, SVD feature selection and training a Neural Network.
'''

## Constants
MODEL_PATH = '../Models/NN_TFIDF_Chi2.joblib'
PREDICT_FILE_PATH = '../Predict creation/Predict.csv'
RESULT_PREDICT_FILE_PATH = 'Predict_result_Chi2.csv'


## Imports

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

'''
Preprocess a CV extract
:parameter
    :param text: string - CV extract
    :param flg_stemm: bool - whether stemming is to be applied
    :param flg_lemm: bool - whether lemmitisation is to be applied
    :param number: bool - whether number removal is to be applied
    :param lst_stopwords: list - list of stopwords to remove
:return
    cleaned CV extract
'''
def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, number=True, lst_stopwords=None):
    # clean --> convert to lowercase and remove punctuations and characters and then strip
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

    # remove number in the text
    if number == True:
        text = re.sub('[0-9]+', '', text)

    # convert from string to list
    lst_text = text.split()

    # remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]

    # Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.SnowballStemmer("english")
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
category = pd.read_csv("../Notebook/categories_string.csv")
label    = pd.read_csv("../Notebook/label.csv")

## Data aquisition
predict = pd.read_csv(PREDICT_FILE_PATH)
data = predict

# Downloading stopword from nltk
print('[INFO] Downloading stopword from nltk')
nltk.download('stopwords')
nltk.download('wordnet')

lst_stopwords = nltk.corpus.stopwords.words("english")

# Preprocessing data (cleaning data)
print('[INFO] Preprocessing data (cleaning data)')
data_clean = data
data_clean["description_clean"] = data_clean["description"].apply(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, number=True, lst_stopwords=lst_stopwords))

# Vectorization with TF-IDF
print('[INFO] Vectorization with TF-IDF')
vectorizerTfidf = feature_extraction.text.TfidfVectorizer(ngram_range=(1,1))
vectorizerTfidf.fit(data_clean["description_clean"])
data_Tfidf = vectorizerTfidf.transform(data_clean["description_clean"])


# Feature selection with Chi2
print('[INFO] Feature selection with Chi2')
y = data_clean["Category"]
X_features_Tfidf = vectorizerTfidf.get_feature_names()

p_value_limit = 0.95
dtf_features_Tfidf = pd.DataFrame()
for cat in np.unique(y):
    chi2, p = feature_selection.chi2(data_Tfidf, y==cat)
    dtf_features_Tfidf = dtf_features_Tfidf.append(pd.DataFrame({"feature":X_features_Tfidf, "score":1-p, "y":cat}))
    dtf_features_Tfidf = dtf_features_Tfidf.sort_values(["y","score"], ascending=[True,False])
    dtf_features_Tfidf = dtf_features_Tfidf[dtf_features_Tfidf["score"]>p_value_limit]
X_features_Tfidf = dtf_features_Tfidf["feature"].unique().tolist()

# Second vectorization with TF-IDF
print('[INFO] Second vectorization with TF-IDF')
vectorizerTfidf = feature_extraction.text.TfidfVectorizer(vocabulary=X_features_Tfidf)
vectorizerTfidf.fit(data_clean["description_clean"])
data_Tfidf = vectorizerTfidf.transform(data_clean["description_clean"])

# Loading model & predictions
print('[INFO] Predictions')
nn_Tfidf = load(MODEL_PATH)

X_features_Tfidf = data_Tfidf

y_predict = nn_Tfidf.predict(X_features_Tfidf)
df_predict = pd.DataFrame(predict)
df_predict['Prediction'] = pd.DataFrame(y_predict)

# Export data in csv file
df_predict.to_csv(RESULT_PREDICT_FILE_PATH)