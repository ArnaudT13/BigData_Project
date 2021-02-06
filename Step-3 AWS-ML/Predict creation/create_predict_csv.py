'''
Description:
    Split the data.json file to obtain the predict part
'''

import pandas as pd
from sklearn.model_selection import train_test_split

data_json_path = "../Notebook/data.json"

print('[INFO] Loading data')
data = pd.read_json(data_json_path)

print('[INFO] Splitting predict dataset')
train, predict = train_test_split(data, test_size=0.2, random_state = 211101)

# Splitting predictions that will be exported
print('[INFO] Create predict csv file')
df_predict = pd.DataFrame(predict)
df_predict.to_csv('Predict.csv')