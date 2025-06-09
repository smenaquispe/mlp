#include <vector>
#include <random>
#include <fstream> 
#include "Layer.h"
#include "ActivationFunction.h"

using namespace std;

#ifndef MULTILAYERPERCEPTRON_H
#define MULTILAYERPERCEPTRON_H

class MultiLayerPerceptron {
public:
    vector<Layer> layers;
    double learningRate = 0.1;
    size_t numLayers = 0;

    MultiLayerPerceptron(
        const vector<int>& layerSizes,
        const vector<ActivationFunction>& activations,
        double learningRate = 0.1
    ) : learningRate(learningRate)
    {
        for (size_t i = 1; i < layerSizes.size(); ++i) {
            layers.emplace_back(Layer(layerSizes[i - 1], layerSizes[i], activations[i - 1]));
        }
        numLayers = layers.size();
    }

    vector<double> forward(const vector<double>& input) {
        vector<double> output = input;
        for (auto& layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }

    void backward(const vector<double>& input, const vector<double>& target) {
        vector<double> output = forward(input);
        this->firstLayerErrors(target, output);
        this->restLayerErrors(target);
        this->updateWeights(input);
    }

    void save_weights(const string& filename) {
        ofstream file(filename);
        if (!file) {
            cerr << "No se pudo abrir el archivo para guardar pesos.\n";
            return;
        }

        for (const auto& layer : layers) {
            for (const auto& neuron_weights : layer.weights) {
                for (double w : neuron_weights)
                    file << w << " ";
                file << "\n";
            }

            for (double b : layer.biases)
                file << b << " ";
            file << "\n"; 
        }

        file.close();
    }

    void load_weights(const string& filename) {
        ifstream file(filename);
        if (!file) {
            cerr << "No se pudo abrir el archivo para cargar pesos.\n";
            return;
        }

        for (auto& layer : layers) {
            for (auto& neuron_weights : layer.weights) {
                for (double& w : neuron_weights)
                    file >> w;
            }

            for (double& b : layer.biases)
                file >> b;
        }

        file.close();
    }

private:
    void firstLayerErrors(const vector<double>& target, const vector<double>& output) {
        for (int i = 0; i < layers[numLayers - 1].numOutputs; ++i) {
            layers[numLayers - 1].errors[i] =
                (output[i] - target[i]) * layers[numLayers - 1].activation.derivative(output[i]);
        }
    }

    void restLayerErrors(const vector<double>& target) {
        for (int l = numLayers - 2; l >= 0; --l) {
            for (int i = 0; i < layers[l].numOutputs; ++i) {
                double sum = 0.0;
                for (int j = 0; j < layers[l + 1].numOutputs; ++j) {
                    sum += layers[l + 1].weights[j][i] * layers[l + 1].errors[j];
                }
                layers[l].errors[i] = sum * layers[l].activation.derivative(layers[l].outputs[i]);
            }
        }
    }

    void updateWeights(const vector<double>& input) {
        for (int l = 0; l < numLayers; ++l) {
            Layer& layer = layers[l];
            const vector<double>& in = l == 0 ? input : layers[l - 1].outputs;

            for (int i = 0; i < layer.numOutputs; ++i) {
                for (int j = 0; j < layer.numInputs; ++j) {
                    layer.weights[i][j] -= learningRate * layer.errors[i] * in[j];
                }
                layer.biases[i] -= learningRate * layer.errors[i];
            }
        }
    }
};

#endif
