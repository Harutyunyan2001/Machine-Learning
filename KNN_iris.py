import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
iris_dataset = load_iris() 

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0) 

from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors=1)    
knn.fit(X_train, y_train) 

X_new = np.array([[5, 2.9, 1, 0.2]]) 
print("форма массива X_new: {}".format(X_new.shape)) 
prediction = knn.predict(X_new) 
print("Прогноз: {}".format(prediction)) 
print("Спрогнозированная метка: {}".format(iris_dataset['target_names'][prediction])) 

y_pred = knn.predict(X_test)
print("Правильность на тестовом наборе: {:.2f}".format(np.mean(y_pred == y_test))) 
print("Правильность на тестовом наборе: {:.2f}".format(knn.score(X_test, y_test))) 





