import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.api.layers import Dense, Input
from keras.api.models import Sequential
from keras.api.optimizers import SGD

def load_data(filename):
    data = pd.read_csv(filename)
    x = data['Altura'].values
    y = data['Peso'].values
    return x, y

def build_model():
    model = Sequential()
    model.add(Dense(1, input_shape=(1,), activation="linear"))
    optimizer = SGD(learning_rate=0.00004)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def fit_model(model, x, y):
    history = model.fit(x, y, epochs=10000, batch_size=len(x), verbose=0)
    return history

def plot_results(history, model, x, y):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.title('Error Cuadratico Medio vs Número de epocas')
    plt.xlabel('Épocas')
    plt.ylabel('Error Cuadratico medio (ECM)')
    plt.grid()
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='datos originales')
    
    x_range = np.linspace(min(x), max(x), 100)
    plt.plot(x_range, model.predict(x_range.reshape(-1, 1)), color='red', label='linea de regresion')    
    plt.title('Altura vs Peso con linea de regresion')
    plt.xlabel('Altura (m)')
    plt.ylabel('Peso (kg)')
    plt.legend()
    plt.grid()
    plt.show()

def estimate_weight(model, height):
    height_normalized = height / 100.0  
    weight = model.predict(np.array([[height_normalized]]))
    print(f'la predicción del peso para una persona con una altura de {height} cm es de {weight[0][0] * 100:.2f} kg')

def execute_program():
    filename = 'altura_peso.csv'
    x, y = load_data(filename) 
    model = build_model()    
    history = fit_model(model, x, y)    
    plot_results(history, model, x, y)    
    estimate_weight(model, 170)

if __name__ == '__main__':
    execute_program()
