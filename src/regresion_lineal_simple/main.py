import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

df = pd.read_csv("src/regrecion_lineal_simple/pa.csv")

x = np.asarray(df['ciclomes']).reshape(-1,1)
y = df['pa']

regr = linear_model.LinearRegression ()
regr. fit (x,y)
y_pred = regr.predict (x)
plt.scatter(x, y, color='black')
plt.plot(x, y_pred, color='blue', linewidth=3) 
plt.ylim(0,1) 
plt.show()
print(regr.predict ([[140]]))