"""
Considerar el archivo pred.csv 
¿Cuál es el coeficiente de determinación del atributo 
'A' utilizando regresión lineal simple? 
(De el resultado con 2 decimales sin aproximar)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score


data = pd.read_csv('src/Tests/pred.csv')
x = np.asarray(data['NO']).reshape(-1, 1)
y = data['A']
print(y)
regr = linear_model.LinearRegression()
regr.fit(x, y)
y_pred = regr.predict(x)
# Datos de rendimiento del modelo
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
variance = explained_variance_score(y, y_pred)
print('RMSE: ', rmse)
print('VARIANCE: ', variance)
print('R2 (Coeficiente de determinacion): ', r2) # El coeficiente de correlacion es la raiz de r cuadrado
plt.scatter(x, y, color='black')
plt.plot(x, y_pred, color='blue', linewidth=3)
# plt.ylim(0, 1) #Establece el limite a graficar
plt.show()
