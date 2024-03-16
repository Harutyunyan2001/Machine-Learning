import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

iris_dataset = load_iris() 
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0) 

model = GaussianNB()
model.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]]) 
print("форма массива X_new: {}".format(X_new.shape)) 
prediction = model.predict(X_new) 
print("Прогноз: {}".format(prediction)) 
print("Спрогнозированная метка: {}".format(iris_dataset['target_names'][prediction])) 

print("train score",model.score(X_train,y_train))
print("test score",model.score(X_test,y_test))
