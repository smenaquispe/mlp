// MLP.h

#include <vector>
#include <random>
#include <algorithm>

#include "ActivationFunction.h"
#include "Layer.h"
#include "ProgressBar.h"

using namespace std;

#ifndef MULTILAYERPERCEPTRON_H
#define MULTILAYERPERCEPTRON_H

class MultiLayerPerceptron
{
public:
    vector<Layer> layers;
    double learningRate = 0.1;
    size_t numLayers = 0;
    string optimizer = "SGD";

    MultiLayerPerceptron(
        const vector<int> &layerSizes,
        const vector<ActivationFunction> &activations,
        double learningRate = 0.1,
        string optimizer = "SGD") : learningRate(learningRate),
                                    optimizer(optimizer)
    {
        for (size_t i = 1; i < layerSizes.size(); ++i)
        {
            layers.emplace_back(Layer(layerSizes[i - 1], layerSizes[i], activations[i - 1]));
        }

        numLayers = layers.size();
    }

    void train(const vector<vector<double>> &trainData,
               const vector<vector<double>> &trainLabels,
               int epochs,
               const vector<vector<double>> &testData = {},
               const vector<vector<double>> &testLabels = {},
               int showEvery = 1)
    {
        for (int epoch = 1; epoch <= epochs; ++epoch)
        {
            cout << "Epoch " << epoch << "/" << epochs << endl;

            double totalLoss = 0.0;
            int correct = 0;

            ProgressBar bar(trainData.size());

            for (size_t i = 0; i < trainData.size(); ++i)
            {
                vector<double> output = forward(trainData[i]);
                backward(trainData[i], trainLabels[i]);

                for (size_t j = 0; j < output.size(); ++j)
                {
                    double diff = output[j] - trainLabels[i][j];
                    totalLoss += diff * diff;
                }

                int predicted = distance(output.begin(), max_element(output.begin(), output.end()));
                int actual = distance(trainLabels[i].begin(), max_element(trainLabels[i].begin(), trainLabels[i].end()));
                if (predicted == actual)
                    correct++;

                bar.update(i + 1);
            }

            bar.finish();

            if (epoch % showEvery == 0)
            {
                double avgLoss = totalLoss / trainData.size();
                double accuracy = static_cast<double>(correct) / trainData.size() * 100.0;
                printf("Loss: %.4f | Accuracy: %.2f%%\n", avgLoss, accuracy);
            }

            if (!testData.empty() && !testLabels.empty())
            {
                double testAccuracy = test(testData, testLabels);
                printf("Test Accuracy: %.2f%%\n", testAccuracy * 100.0);
            }
        }
    }

    double test(const vector<vector<double>> &testData,
                const vector<vector<double>> &testLabels)
    {
        int correct = 0;

        for (size_t i = 0; i < testData.size(); ++i)
        {
            vector<double> output = forward(testData[i]);

            int predicted = distance(output.begin(), max_element(output.begin(), output.end()));
            int actual = distance(testLabels[i].begin(), max_element(testLabels[i].begin(), testLabels[i].end()));

            if (predicted == actual)
                correct++;
        }

        return static_cast<double>(correct) / testData.size();
    }

    void save_weights(const std::string &filename)
    {
        ofstream out(filename);
        if (!out.is_open())
        {
            cerr << "Error abriendo archivo para guardar pesos.\n";
            return;
        }

        for (const Layer &layer : layers)
        {
            for (const auto &row : layer.weights)
            {
                for (double w : row)
                {
                    out << w << " ";
                }
                out << "\n";
            }
            for (double b : layer.biases)
            {
                out << b << " ";
            }
            out << "\n#\n";
        }

        out.close();
    }

    void load_weights(const std::string &filename)
    {
        ifstream in(filename);
        if (!in.is_open())
        {
            cerr << "Error abriendo archivo para cargar pesos.\n";
            return;
        }

        for (Layer &layer : layers)
        {
            for (auto &row : layer.weights)
            {
                for (double &w : row)
                {
                    in >> w;
                }
            }
            for (double &b : layer.biases)
            {
                in >> b;
            }

            string line;
            getline(in, line);
            while (line != "#" && getline(in, line))
            {
            }
        }

        in.close();
    }

private:
    vector<double> forward(const vector<double> &input)
    {
        vector<double> output = input;
        for (auto &layer : layers)
        {
            output = layer.forward(output);
        }
        return output;
    }

    void backward(
        const vector<double> &input,
        const vector<double> &target)

    {
        vector<double> output = forward(input);
        this->firstLayerErrors(target, output);
        this->restLayerErrors(target);
        this->updateWeights(input);
    }

    void firstLayerErrors(
        const vector<double> &target,
        const vector<double> &output)
    {
        for (int i = 0; i < layers[numLayers - 1].numOutputs; ++i)
        {
            layers[numLayers - 1].errors[i] = (output[i] - target[i]) * layers[numLayers - 1].activation.derivative(output[i]);
        }
    }

    void restLayerErrors(
        const vector<double> &target)
    {
        for (int l = numLayers - 2; l >= 0; --l)
        {
            for (int i = 0; i < layers[l].numOutputs; ++i)
            {
                double sum = 0.0;
                for (int j = 0; j < layers[l + 1].numOutputs; ++j)
                {
                    sum += layers[l + 1].weights[j][i] * layers[l + 1].errors[j];
                }
                layers[l].errors[i] = sum * layers[l].activation.derivative(layers[l].outputs[i]);
            }
        }
    }

    void updateWeights(const vector<double> &input)
    {
        for (int l = 0; l < numLayers; ++l)
        {
            Layer &layer = layers[l];
            const vector<double> &in = (l == 0 ? input : layers[l - 1].outputs);

            for (int i = 0; i < layer.numOutputs; ++i)
            {
                for (int j = 0; j < layer.numInputs; ++j)
                {
                    layer.weights[i][j] -= learningRate * layer.errors[i] * in[j];
                }
                layer.biases[i] -= learningRate * layer.errors[i];
            }
        }
    }
};

#endif