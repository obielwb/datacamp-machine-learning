# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1JOXpWsam-cahK0KGZLy3NL6wIQRx2sQI
"""

# Construindo um modelo de classificação usando algoritmo de KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")
iris = datasets.load_iris()

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(iris['data'], iris['target'])

print(iris['data'].shape) # Exibe o tamanho dos dados

print(iris['target'].shape) # Exibe apenas o tamanho da coluna

X_new = np.array([[5.6, 2.8, 3.9, 1.1],
                  [5.7, 2.6, 3.8, 1.3],
                  [4.7, 3.2, 1.3, 0.2]])

# Com base nos dados rotulados anteriores
# preveja o rótulo de novos dados não rotulados
prediction = knn.predict(X_new) 
print('Prediction: {}'.format(prediction))