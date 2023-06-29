""" 
Considerar el archivo pred.csv donde A, B, C y D son variables de entrada y
E es la variable de salida, también el algoritmo y parámetros por defecto del 
clasificador GaussianNB de scikit learn (no utilizar LabelEncoder) 
¿Cuál es la predicción (N o P) para los valores de entrada [100,0,100,False]? 
"""

import pandas as pd
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('src/Tests/pred.csv')
x = data[['A', 'B', 'C', 'D']]
y = data['E']

model = GaussianNB()
model.fit(x,y)

input = [[100,0,100,False]]
predicted = model.predict(input)
print("Predicted Value: ",predicted)