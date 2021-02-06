import pandas as pd
from sklearn.model_selection import train_test_split

print('[INFO] Loading data')
data = pd.read_json("../Notebook/data.json")

print('[INFO] Splitting predict dataset')
train, predict = train_test_split(data, test_size=0.2, random_state = 211101)

# Splitting predictions that will be exported
print('[INFO] Create predict csv file')
df_predict = pd.DataFrame(predict)
df_predict.to_csv('Predict.csv')