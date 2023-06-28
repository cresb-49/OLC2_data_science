from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import numpy as np
import random

# Paso 1: informacion de entrenamiento
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
y = [0, 1, 2, 3, 6, 6, 8, 9, 13, 17, 19, 20, 21, 24, 25, 32, 34, 36, 39, 46, 47, 50, 61, 70, 74, 80]

x = np.asarray(x)
y = np.asarray(y)

x = x[:,np.newaxis]
y = y[:,np.newaxis]

plt.scatter(x,y)

# plt.show()

# Paso 2: Preparacion de datos

nb_degree = 3 # Grado del polinomio

polynomia_features = PolynomialFeatures(degree= nb_degree)

X_TRANSF = polynomia_features.fit_transform(x)

# Paso 3: Definicion y entrenamiento del modelo

model = LinearRegression()
model.fit(X_TRANSF,y)

# Paso 4: Calcular el sesgo (bias) y la varianza

Y_NEW = model.predict(X_TRANSF)
rmse = np.sqrt(mean_squared_error(y,Y_NEW))
r2 = r2_score(y,Y_NEW)

# Varian los resultados con la precentacion ya que no 
# pude recuperar todos los demas datos
print('RMSE: ',rmse)
print('R2: ',r2)

# Paso 5: Prediccion

x_new_min = 0.0
x_new_max= 50.0

X_NEW = np.linspace(x_new_min,x_new_max,50)
X_NEW = X_NEW[:,np.newaxis]

X_NEW_TRANSF = polynomia_features.fit_transform(X_NEW)

Y_NEW = model.predict(X_NEW_TRANSF)

plt.plot(X_NEW,Y_NEW, color='coral',linewidth=3)

plt.grid()
plt.xlim(x_new_min,x_new_max)
plt.ylim(0,1000)

tittle = 'Degree = {}; RMSE = {}; R2 ={}'.format(nb_degree,round(rmse,2),round(r2,2))

plt.title('Polynomial Linear Regression using scikit-learn and python 3 \n',fontsize=10)
plt.xlabel('x')
plt.ylabel('y')

plt.savefig('polynomial_linear_regression.png',bbox_inches='tight')
plt.show()