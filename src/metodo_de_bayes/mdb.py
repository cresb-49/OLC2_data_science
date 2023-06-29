from sklearn import preprocessing
import matplotlib.pyplot as plt
#Import del modelo Gaussiano de Bayes
from sklearn.naive_bayes import GaussianNB

# Datacreation
outlook = ['sunny', 'sunny', 'overcast', 'rain', 'rain', 'rain', 'overcast',
           'sunny', 'sunny', 'rain', 'sunny', 'overcast', 'overcast', 'rain']
temperature = ['hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool',
               'mild', 'cool', 'mild', 'mild', 'mild', 'hot', 'mild']
humidity = ['high', 'high', 'high', 'high', 'normal', 'normal', 'normal',
            'high', 'normal', 'normal', 'normal', 'high', 'normal', 'high']
windy = ['false', 'true', 'false', 'false', 'false', 'true', 'true',
         'false', 'false', 'false', 'true', 'true', 'false', 'true']
play = ['N', 'N', 'P', 'P', 'P', 'N', 'P', 'N', 'P', 'P', 'P', 'P', 'P', 'N']

# Creacion de labelEncoder
le = preprocessing.LabelEncoder()

# Conversion de las etiquetas de caracteres en numeros
outlook_econded = le.fit_transform(outlook)
temperature_encoded = le.fit_transform(temperature)
humidity_encoded = le.fit_transform(humidity)
windy_encoded = le.fit_transform(windy)
label = le.fit_transform(play)

print('outlook: ', outlook_econded)
print('temp: ', temperature_encoded)
print('humidity: ', humidity_encoded)
print('windy: ', windy_encoded)
print('PLAY: ', label)

# Combinacion de los atributos dentro de una lista de tuplas
features=list(zip(outlook_econded,temperature_encoded,humidity_encoded,windy_encoded))
print(features)

# Creacion del Clasificador Gaussiano
model = GaussianNB()

# Entrenamiento del modelos utilizando los datos preparados
model.fit(features,label)

# Salida predictiva
predicted =model.predict([[2,1,0,0]]) # sunny, hot, high, false
print("Predicted Value: ",predicted)