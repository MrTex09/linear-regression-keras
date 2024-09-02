# Proyecto de Regresión Lineal con Keras

## Requisitos

- Python 3.x
- Librerías:
  - pandas
  - numpy
  - matplotlib
  - tensorflow (Keras está incluido en TensorFlow)

## Instalación de Dependencias

1. **Crear y activar un entorno virtual (opcional pero recomendado):**

   ```bash
   python -m venv env
   .\env\Scripts\activate  # Windows
   source env/bin/activate # Linux/Mac
   ```

2. **Crear un archivo `requirements.txt` con las dependencias necesarias:**
   ```
   pandas
   numpy
   matplotlib
   tensorflow
   ```
    instala las dependencias ejecutando

   ```bash
   pip install -r requirements.txt
   ```

## Uso

1. **Coloca el archivo `altura_peso.csv` en el mismo directorio que `main.py`.**

2. **Ejecuta el script:**

   ```bash
   python main.py
   ```

   El script leerá los datos del archivo CSV entrenara un modelo de regresión lineal y visualizará los resultados en un gráfico.

