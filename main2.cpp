#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <random>

#include "ActivationFunction.h"
#include "MLP.h"
#include "softmax.h"

using namespace std;

ActivationFunction relu(
    [](double x)
    { return x > 0 ? x : 0; },
    [](double x)
    { return x > 0 ? 1 : 0; });

ActivationFunction linear(
    [](double x)
    { return x; },
    [](double x)
    { return 1; });

void load_csv(const string &filename, vector<vector<double>> &inputs, vector<vector<double>> &targets)
{
    ifstream file(filename);
    string line;

    getline(file, line); // skip header

    while (getline(file, line))
    {
        stringstream ss(line);
        string val;
        vector<double> row;

        getline(ss, val, ',');
        int label = stoi(val);
        vector<double> target(10, 0.0);
        target[label] = 1.0;

        while (getline(ss, val, ','))
        {
            row.push_back(stoi(val) / 255.0);
        }

        inputs.push_back(row);
        targets.push_back(target);
    }
}

void reduce_dataset_balanced(vector<vector<double>> &inputs,
                             vector<vector<double>> &targets,
                             double fraction = 0.5)
{
    vector<vector<int>> class_indices(10); // 10 clases (0–9)

    // Agrupar índices por clase
    for (size_t i = 0; i < targets.size(); ++i)
    {
        int label = distance(targets[i].begin(), max_element(targets[i].begin(), targets[i].end()));
        class_indices[label].push_back(i);
    }

    // Nuevos vectores reducidos
    vector<vector<double>> reduced_inputs, reduced_targets;
    random_device rd;
    mt19937 gen(rd());

    for (int label = 0; label < 10; ++label)
    {
        auto &indices = class_indices[label];
        shuffle(indices.begin(), indices.end(), gen);

        size_t count = static_cast<size_t>(indices.size() * fraction);

        for (size_t j = 0; j < count; ++j)
        {
            int idx = indices[j];
            reduced_inputs.push_back(inputs[idx]);
            reduced_targets.push_back(targets[idx]);
        }
    }

    // Sobrescribir con el subconjunto reducido
    inputs = move(reduced_inputs);
    targets = move(reduced_targets);
}

int main()
{

    vector<vector<double>> inputs, targets;
    vector<vector<double>> inputsTest, targetsTest;

    cout << "Loading MNIST dataset..." << endl;
    load_csv("mnist_train.csv", inputs, targets);
    cout << "Loaded " << inputs.size() << " samples." << endl;

    reduce_dataset_balanced(inputs, targets, 0.5);

    cout << "Loading MNIST test dataset..." << endl;
    load_csv("mnist_test.csv", inputsTest, targetsTest);
    cout << "Loaded " << inputsTest.size() << " test samples." << endl;

    MultiLayerPerceptron mlp({784, 128, 10}, {relu, relu}, 0.01, "SGD");

    mlp.train(inputs, targets, 10, inputsTest, targetsTest, 1);
    mlp.save_weights("mlp_weights.txt");

    return 0;
}