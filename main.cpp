#include <vector>
#include <iostream>
#include "MLP.h"
#include "ActivationFunction.h"


ActivationFunction sigmoid(
    [](double x) { return 1.0 / (1.0 + exp(-x)); },
    [](double y) { return y * (1.0 - y); }
);

ActivationFunction relu(
    [](double x) { return x > 0 ? x : 0; },
    [](double x) { return x > 0 ? 1 : 0; }
);

ActivationFunction tanh_act(
    [](double x) { return tanh(x); },
    [](double y) { return 1 - y * y; }
);

ActivationFunction step(
    [](double x) { return x >= 0 ? 1 : 0; },
    [](double y) { return 0; }
);


using namespace std;

int main() {
    MultiLayerPerceptron mlp({2, 3, 1}, {sigmoid, sigmoid}, 0.4); // 2 entradas, 1 capa oculta de 3 neuronas, 1 salida

    vector<vector<double>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };

    vector<vector<double>> targets = {
        {0}, {1}, {1}, {0}
    };

    for (int epoch = 0; epoch < 100000; ++epoch) {
        for (int i = 0; i < 4; ++i)
            mlp.backward(inputs[i], targets[i]);
    }

    for (int i = 0; i < 4; ++i) {
        vector<double> output = mlp.forward(inputs[i]);
        cout << inputs[i][0] << " XOR " << inputs[i][1]
                  << " = " << output[0] << endl;
    }

    return 0;
}
