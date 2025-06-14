#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <random>
#include "MLP.h"
#include "ActivationFunction.h"
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

void processBatch(const vector<pair<vector<double>, vector<double>>> &batch,
                  MultiLayerPerceptron &mlp, double &total_error)
{
    double local_error = 0.0;

    cout << "Processing batch of size: " << batch.size() << endl;

    for (const auto &data : batch)
    {
        vector<double> output = mlp.forward(data.first);
        vector<double> softmax_output = softmax(output);
        mlp.backward(data.first, data.second);

        for (size_t j = 0; j < softmax_output.size(); ++j)
        {
            local_error += -data.second[j] * log(max(softmax_output[j], 1e-15));
        }
    }

    cout << "Local error for batch: " << local_error << endl;
    total_error += local_error;
}

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

int main()
{
    vector<vector<double>> inputs, targets;
    load_csv("mnist_train.csv", inputs, targets);

    MultiLayerPerceptron mlp({784, 256, 128, 10}, {relu, relu, linear}, 0.001, "Adam");
    double best_error = numeric_limits<double>::max();
    int patience = 3;
    int wait = 0;

    ofstream curve("learning_curve.csv");
    curve << "epoch,error\n";

    int epochs = 50;
    int batch_size = 100;

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double total_error = 0.0;
        cout << "Epoch " << epoch + 1 << "/" << epochs << endl;

        vector<size_t> indices(inputs.size());
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), mt19937{random_device{}()});

        for (size_t i = 0; i < inputs.size(); i += batch_size)
        {
            size_t end = min(i + batch_size, inputs.size());
            vector<pair<vector<double>, vector<double>>> batch;

            for (size_t j = i; j < end; ++j)
            {
                batch.emplace_back(inputs[indices[j]], targets[indices[j]]);
            }

            processBatch(batch, mlp, total_error);
        }

        total_error /= inputs.size();
        cout << "Epoch " << epoch + 1 << " Error: " << total_error << endl;
        curve << epoch + 1 << "," << total_error << "\n";

        // Early stopping
        if (total_error < best_error)
        {
            best_error = total_error;
            wait = 0;
            mlp.save_weights("best_weights.txt");
        }
        else
        {
            wait++;
            if (wait >= patience)
            {
                cout << "Early stopping at epoch " << epoch + 1 << endl;
                break;
            }
        }
    }

    curve.close();

    return 0;
}
