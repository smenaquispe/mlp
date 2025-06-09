#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include "MLP.h"
#include "ActivationFunction.h"

using namespace std;

// Definir funciones de activación
ActivationFunction sigmoid(
    [](double x) { return 1.0 / (1.0 + exp(-x)); },
    [](double y) { return y * (1.0 - y); }
);

int argmax(const vector<double>& v) {
    int idx = 0;
    double max_val = v[0];
    for (int i = 1; i < v.size(); ++i) {
        if (v[i] > max_val) {
            max_val = v[i];
            idx = i;
        }
    }
    return idx;
}

vector<double> parseLine(const string& line) {
    vector<double> pixels;
    stringstream ss(line);
    string item;

    while (getline(ss, item, ',')) {
        pixels.push_back(stoi(item) / 255.0); // normalizar
    }

    return pixels;
}

int main() {
    // MLP: 784 entradas, 1 oculta de 64, 10 salidas
    MultiLayerPerceptron mlp({784, 64, 10}, {sigmoid, sigmoid}, 0.1);
    
    // Cargar pesos previamente entrenados
    mlp.load_weights("mlp_weights.txt");

    ifstream file("mnist_test.csv");
    if (!file.is_open()) {
        cerr << "No se pudo abrir mnist_test.csv" << endl;
        return 1;
    }

    string line;
    int correctos = 0;

    while (getline(file, line)) {
        vector<double> entrada = parseLine(line);
        vector<double> salida = mlp.forward(entrada);
        int predicho = argmax(salida);

        cout << "Etiqueta Predicho: " << predicho << endl;

    }


    file.close();
    return 0;
}
