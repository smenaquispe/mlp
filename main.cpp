#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "MLP.h"
#include "ActivationFunction.h"
#include "softmax.h"

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

ActivationFunction linear(
    [](double x) { return x; },
    [](double x) { return 1; }
);

double cross_entropy_loss(const vector<double>& y_pred, const vector<double>& y_true) {
    double loss = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        loss -= y_true[i] * log(y_pred[i] + 1e-15); // evitar log(0)
    }
    return loss;
}


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

    MultiLayerPerceptron mlp({784, 256, 128, 10}, {relu, relu, linear}, 0.1, OptimizerType::Adam );

    ofstream curve("learning_curve.csv");
    curve << "epoch,error\n";

    int epochs = 10;

    for (int epoch = 0; epoch < epochs; ++epoch) {
    double total_loss = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            vector<double> logits = mlp.forward(inputs[i]);
            vector<double> probs = softmax(logits);



            mlp.backward_from_softmax(inputs[i], targets[i], probs);

            total_loss += cross_entropy_loss(probs, targets[i]);
        }

        cout << "Epoch " << epoch + 1 << " - Loss: " << total_loss / inputs.size() << endl;
    }


    // guardamos pesos para usarlo luego
    mlp.save_weights("mlp_weights.txt");

    curve.close();

    return 0;
}
