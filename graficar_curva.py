import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("learning_curve.csv")

df['accuracy'] = 1 - df['error']

plt.figure(figsize=(10, 6))

plt.plot(df['epoch'], df['error'], label='Pérdida (Error)', color='red', marker='o')

plt.plot(df['epoch'], df['accuracy'], label='Precisión estimada', color='blue', marker='x')

plt.title('Curva de Pérdida y Precisión durante el Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig('learning_curve.png')
