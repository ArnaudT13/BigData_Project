'''
Description:
    Split the data.json file to obtain the predict part
'''

## Constants
DATA_JSON_PATH = "../Data/data.json"
PREDICT_FILE = "Predict.csv"


## Imports
import pandas as pd
from sklearn.model_selection import train_test_split


## Load and split dataset
print('[INFO] Loading data')
data = pd.read_json(DATA_JSON_PATH)

print('[INFO] Splitting predict dataset')
train, predict = train_test_split(data, test_size=0.2, random_state = 211101)

## Export prediction
print('[INFO] Create predict csv file')
df_predict = pd.DataFrame(predict)
df_predict.to_csv(PREDICT_FILE)