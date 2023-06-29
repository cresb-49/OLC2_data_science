'''
Considerar el archivo pred.csv donde A, B, C y D son variables de entrada y E es la variable de 
salida, también el algoritmo y parámetros por defecto de árboles de decisión de scikit learn 
(no utilizar LabelEncoder) ¿En cuántos nodos hoja se clasifica la clase N?
'''

from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

data = pd.read_csv('src/Tests/pred.csv')
# VARIABLES DE ENTRDA
column_a = np.asarray(data['A'])
column_b = np.asarray(data['B'])
column_c = np.asarray(data['C'])
column_d = np.asarray(data['D'])
# VARIABLES DE SALIDA
column_e = np.asarray(data['E'])

print('A: ',column_a)
print('B: ',column_b)
print('C: ',column_c)
print('D: ',column_d)
print('E: ',column_e)

features = list(zip(column_a,column_b,column_c,column_d))
print(features)
clf = DecisionTreeClassifier().fit(features,column_e)
plot_tree(clf,filled=True)
plt.show()