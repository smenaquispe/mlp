#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
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

void load_csv(const string& filename, vector<vector<double>>& inputs, vector<vector<double>>& targets) {
    ifstream file(filename);
    string line;

    getline(file, line); 

    while (getline(file, line)) {
        stringstream ss(line);
        string val;
        vector<double> row;
        
        getline(ss, val, ','); 
        int label = stoi(val);
        vector<double> target(10, 0.0);
        target[label] = 1.0;  

        while (getline(ss, val, ',')) {
            row.push_back(stoi(val) / 255.0); 
        }

        inputs.push_back(row);
        targets.push_back(target);
    }
}

int main() {
    
    vector<vector<double>> inputs, targets;
    load_csv("mnist_train.csv", inputs, targets);

    MultiLayerPerceptron mlp({784, 64, 10}, {sigmoid, sigmoid}, 0.1);
    
    ofstream curve("learning_curve.csv");
    curve << "epoch,error\n";

    int epochs = 40;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_error = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            vector<double> output = mlp.forward(inputs[i]);
            mlp.backward(inputs[i], targets[i]);

            for (size_t j = 0; j < output.size(); ++j) {
                double diff = targets[i][j] - output[j];
                total_error += diff * diff;
            }
        }

        total_error /= inputs.size();
        cout << "Epoch " << epoch + 1 << " Error: " << total_error << endl;
        curve << epoch + 1 << "," << total_error << "\n";
    }

    // guardamos pesos para usarlo luego
    mlp.save_weights("mlp_weights.txt");

    curve.close();

    return 0;
}
