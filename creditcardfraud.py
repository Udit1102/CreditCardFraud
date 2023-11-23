import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading and inspection of data

transactions = pd.read_csv("transactions.csv")
#print(transactions.info())
#print(transactions.head())
transactions.columns = transactions.columns.str.lower()

#data cleaning and insights

#print(transactions[["step", "type"]][:5])
#print(transactions.type.value_counts())
#print(np.mean(transactions.amount))
#print(np.median(transactions.amount))
#print(np.var(transactions.amount))

#splitting the data 

features = transactions[["amount", "ispayment", "ismovement", "accountdiff"]]
label = transactions["isfraud"]
features_train, features_test, label_train, label_test = train_test_split(features, label, test_size = 0.3)

#standardization of features

scaler = StandardScaler()
scaler.fit(features)
features = scaler.transform(features)

#trainning the model

model = LogisticRegression()
model.fit(features_train, label_train)

#checking the accuracy

print(model.score(features_train, label_train))
print(model.score(features_test, label_test))
print(model.coef_)

predictions = model.predict(features_test)
#print(predictions)

#sample data

sample = np.array([[123456.78, 0.0, 1.0, 54670.1]])
#print(sample)
scaler.fit(sample)
sample = scaler.transform(sample)
sample_predictions = model.predict(sample)
print(sample_predictions)