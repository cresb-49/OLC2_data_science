'''
Considerar el archivo pred.csv ¿Cuál es el valor de r cuadrado del atributo B utilizando regresión polinomial de grado 2?
'''
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

df = pd.read_csv('src\Tests\pred.csv')
x = np.asarray(df['B'])
y = np.asarray(df['NO'])

x = x[:, np.newaxis]
y = y[:, np.newaxis]

plt.scatter(x, y)

nb_degree = 2  # Grado del polinomio

polynomia_features = PolynomialFeatures(degree=nb_degree)

X_TRANSF = polynomia_features.fit_transform(x)

model = LinearRegression()
model.fit(X_TRANSF, y)

Y_NEW = model.predict(X_TRANSF)
rmse = np.sqrt(mean_squared_error(y, Y_NEW))
r2 = r2_score(y, Y_NEW)
variance = explained_variance_score(y, Y_NEW)

print('RMSE: ', rmse)
print('R2: ', r2)
print('VARIANCE: ', variance)
