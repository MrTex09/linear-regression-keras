import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

def load_data_from_csv(file_path):
    data = pd.read_csv(file_path)
    heights = data['Altura'].values
    weights = data['Peso'].values
    return heights, weights

def build_linear_regression_model():
    model = Sequential()
    model.add(Dense(1, input_shape=(1,), activation="linear"))
    optimizer = SGD(learning_rate=0.00001)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def train_model(model, heights, weights):
    epochs = 10000
    batch_size = len(heights)
    training_history = model.fit(heights, weights, epochs=epochs, batch_size=batch_size, verbose=0)
    return training_history

def plot_training_results(training_history, model, heights, weights):
    plt.figure(figsize=(10, 6))
    plt.plot(training_history.history['loss'])
    plt.title('error Cuadratico Medio vs Numero de epocas')
    plt.xlabel('Épocas')
    plt.ylabel('Error Cuadrático Medio (ECM)')
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.scatter(heights, weights, label='Datos Originales')    
    x_range = np.linspace(min(heights), max(heights), 100)
    y_pred = model.predict(x_range.reshape(-1, 1))
    
    plt.plot(x_range, y_pred, color='red', label='linea de regresion')
    plt.title('Altura vs Peso con Línea de Regresion')
    plt.xlabel('Altura')
    plt.ylabel('Peso')
    plt.legend()
    plt.show()

def predict_weight_for_height(model, height_cm):
    weight_prediction = model.predict(np.array([[height_cm]]))
    print(f'La predicción del peso para una persona con una altura de {height_cm} cm es de {weight_prediction[0][0]:.2f} kg')

def run_linear_regression_program():
    file_path = 'altura_peso.csv'
    heights, weights = load_data_from_csv(file_path)    
    model = build_linear_regression_model()    
    training_history = train_model(model, heights, weights)    
    plot_training_results(training_history, model, heights, weights)    
    predict_weight_for_height(model, 170)

if __name__ == '__main__':
    run_linear_regression_program()
