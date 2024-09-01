import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input

data = pd.read_csv('altura_peso.csv')
print(data.columns)

x = data['Altura'].values
y = data['Peso'].values
x = x.reshape(-1, 1)

model = Sequential()
model.add(Input(shape=(1,))) 
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(x, y, epochs=100)
predicciones = model.predict(x)

plt.scatter(x, y, color='blue', label='Datos originales')
plt.plot(x, predicciones, color='red', label='Predicción')
plt.xlabel('Altura')
plt.ylabel('Peso')
plt.title('Regresión Lineal - Altura vs Peso')
plt.legend()
plt.show()