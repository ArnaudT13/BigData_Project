'''
Description:
    Performs NLP on CV/resume description prediction by using Word2Vec with Neural Network model.

    Warning !! The Word2Vec model must be downloaded here : https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz and place in the same script location.
               The model should be also in the same location.
'''
## Imports
# for data
import pandas as pd
import numpy as np

# machine learning
from joblib import dump, load


## Data aquisition
predict = pd.read_csv("predict.csv")

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


# Tokenization
print('[INFO] Tokenization')
data_tokenized = data_clean.apply(lambda r: w2v_tokenize_text(r['description_clean']), axis=1).values

# Averaging
print('[INFO] Averaging')
X_word_average = word_averaging_list(wv, data_tokenized)

# Get model
model_word2vec_nn = load('Models/NN_Word2Vec.joblib')

# Create dataset prediction
y_predict = model_word2vec_nn.predict(X_word_average)
df_predict = pd.DataFrame(predict)
df_predict['Prediction'] = pd.DataFrame(y_predict)

# Export data in csv file
df_predict.to_csv('Predict_result.csv')