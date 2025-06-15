import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer

# Cargar datasets
train_df = pd.read_csv('mnist_train.csv')
test_df = pd.read_csv('mnist_test.csv')

# Separar inputs y targets
X_train = train_df.iloc[:, 1:].values / 255.0  # Normalizar
y_train = train_df.iloc[:, 0].values

X_test = test_df.iloc[:, 1:].values / 255.0
y_test = test_df.iloc[:, 0].values

# Mostrar un ejemplo para verificar
print("Ejemplo:")
print("Label:", y_train[0])
print("Pixel data (first 10):", X_train[0][:10])

# Entrenar clasificador simple
clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=20, solver='adam', random_state=42)
clf.fit(X_train, y_train)

# Evaluar
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy en test: {acc * 100:.2f}%")
