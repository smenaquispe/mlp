import matplotlib.pyplot as plt
import pandas as pd

df_train = pd.read_csv("mnist_train.csv")
df_test = pd.read_csv("mnist_test.csv")

# Mostrar una imagen del set de entrenamiento
pixels_train = df_train.iloc[0, 1:].values.reshape(28, 28)
plt.imshow(pixels_train, cmap='gray')
plt.title(f"Label: {df_train.iloc[0, 0]}")
plt.show()

# Mostrar una imagen del set de prueba
pixels_test = df_test.iloc[0, 1:].values.reshape(28, 28)
plt.imshow(pixels_test, cmap='gray')
plt.title(f"Label: {df_test.iloc[0, 0]}")
plt.show()
