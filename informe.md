# Informe de Regresión Lineal con Keras

  **Entrenamiento del Modelo:**
   - El modelo se entrenó utilizando 10,000 épocas y un tamaño de batch igual al número total de datos. El optimizador utilizado fue `SGD` con una tasa de aprendizaje de 0.0004 y la función de pérdida fue el error cuadrático medio (MSE).
   - Los valores finales del peso (w) y el sesgo (b) del modelo se obtuvieron después del entrenamiento: `w = X.XX`, `b = Y.YY`.

## Interpretación de Resultados
El modelo de regresión lineal implementado es capaz de capturar la relación lineal entre la altura y el peso de las personas. La disminución constante del error cuadrático medio durante el entrenamiento sugiere que el modelo está aprendiendo correctamente esta relación. La recta de regresión ajustada se alinea bien con los datos originales, lo que indica una buena capacidad de ajuste para este conjunto de datos.

