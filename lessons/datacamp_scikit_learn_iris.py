# -*- coding: utf-8 -*-
"""Datacamp Scikit-Learn-Iris

Colaboratory file
    https://colab.research.google.com/drive/1acZmXNYfpfKGED1LJF7xXYFHH5dbrSuw
"""

from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")
# Armazena o conjunto de dados na variavel iris
iris = datasets.load_iris()
type(iris)

print(iris.keys())
type(iris.data), type(iris.target)
iris.data.shape

# Tipos de flores
iris.target_names

# Exploratory data analysis (EDA)
x = iris.data
y = iris.target
df = pd.DataFrame(x, columns=iris.feature_names)
print(df.head())

# Visual EDA
_ = pd.plotting.scatter_matrix(df, c = y, figsize = [8,8], s = 150, marker = 'd')