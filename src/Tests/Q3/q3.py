"""
Considerar el archivo pred.csv ¿Cuál es la predicción del atributo C para un correlativo 30 
utilizando regresión lineal simple? (De el resultado con 2 decimales sin aproximar)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('src/Tests/pred.csv')
x = np.asarray(data['NO']).reshape(-1, 1)
y = data['C']
# Set del modelo
regr = linear_model.LinearRegression()
regr.fit(x, y)
y_pred = regr.predict(x)
# Datos de rendimiento de la regrecion
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
print('RMSE: ', rmse)
print('R2: ', r2)
# Prediccion de un nuevo valor
correlative = 30
prediction_correlative = regr.predict([[correlative]])
print('Predicción: ', prediction_correlative)
# Graficado del modelo
plt.scatter(x, y, color='black')
plt.plot(x, y_pred, color='blue', linewidth=3)
plt.show()
