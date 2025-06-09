#ifndef MULTILAYERPERCEPTRON_H
#define MULTILAYERPERCEPTRON_H

#include <vector>
#include <random>
#include <fstream>
#include <cmath>
#include <iostream>
#include "Layer.h"
#include "ActivationFunction.h"
#include "softmax.h"

using namespace std;

class MultiLayerPerceptron {
public:
    vector<Layer> layers;
    double learningRate;
    size_t numLayers;
    string optimizer;

    double beta_rmsprop;
    vector<vector<vector<double>>> cache_rmsprop;

    double beta1_adam;
    double beta2_adam;
    vector<vector<vector<double>>> m_adam;
    vector<vector<vector<double>>> v_adam;
    int t_adam;

    MultiLayerPerceptron(
        const vector<int>& layerSizes,
        const vector<ActivationFunction>& activations,
        double learningRate = 0.001,
        string optimizer = "Adam"
    ) : learningRate(learningRate), optimizer(optimizer), beta_rmsprop(0.9), beta1_adam(0.9), beta2_adam(0.999), t_adam(0) {
        for (size_t i = 1; i < layerSizes.size(); ++i) {
            layers.emplace_back(Layer(layerSizes[i - 1], layerSizes[i], activations[i - 1]));
        }
        numLayers = layers.size();

        if (optimizer == "RMSProp") {
            initializeCacheRMSProp();
        }

        if (optimizer == "Adam") {
            initializeMomentsAdam();
        }
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
        firstLayerErrors(target, output);
        restLayerErrors(target);
        updateWeights(input);
    }

    void save_weights(const string& filename) {
        ofstream file(filename);
        if (!file) {
            cerr << "No se pudo abrir el archivo para guardar pesos.\n";
            return;
        }

        for (const auto& layer : layers) {
            for (const auto& neuron_weights : layer.weights) {
                for (double w : neuron_weights) {
                    file << w << " ";
                }
                file << "\n";
            }

            for (double b : layer.biases) {
                file << b << " ";
            }
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
                for (double& w : neuron_weights) {
                    file >> w;
                }
            }

            for (double& b : layer.biases) {
                file >> b;
            }
        }

        file.close();
    }

private:
    void initializeCacheRMSProp() {
        for (auto& layer : layers) {
            vector<vector<double>> layerCache;
            for (auto& neuron_weights : layer.weights) {
                vector<double> neuronCache(neuron_weights.size(), 0.0);
                layerCache.push_back(neuronCache);
            }
            cache_rmsprop.push_back(layerCache);
        }
    }

    void initializeMomentsAdam() {
        for (auto& layer : layers) {
            vector<vector<double>> layerM;
            vector<vector<double>> layerV;
            for (auto& neuron_weights : layer.weights) {
                vector<double> neuronM(neuron_weights.size(), 0.0);
                vector<double> neuronV(neuron_weights.size(), 0.0);
                layerM.push_back(neuronM);
                layerV.push_back(neuronV);
            }
            m_adam.push_back(layerM);
            v_adam.push_back(layerV);
        }
    }

        void firstLayerErrors(const vector<double>& target, const vector<double>& output) {
            vector<double> softmax_output = softmax(output);
            for (int i = 0; i < layers[numLayers - 1].numOutputs; ++i) {
                layers[numLayers - 1].errors[i] = (softmax_output[i] - target[i]);
            }
        }

        void restLayerErrors(const vector<double>& target) {
            for (int l = numLayers - 2; l >= 0; --l) {
                for (int i = 0; i < layers[l].numOutputs; ++i) {
                    double sum = 0.0;
                    for (int j = 0; j < layers[l + 1].numOutputs; ++j) {
                        sum += layers[l + 1].weights[j][i] * layers[l + 1].errors[j];
                    }
                    layers[l].errors[i] = sum * ((layers[l].outputs[i] > 0) ? 1.0 : 0.0);
                }
            }
        }

    void updateWeights(const vector<double>& input) {
        t_adam++;
        for (int l = 0; l < numLayers; ++l) {
            Layer& layer = layers[l];
            const vector<double>& in = l == 0 ? input : layers[l - 1].outputs;

            for (int i = 0; i < layer.numOutputs; ++i) {
                for (int j = 0; j < layer.numInputs; ++j) {
                    double gradient = layer.errors[i] * in[j];
                    if (optimizer == "SGD") {
                        layer.weights[i][j] -= learningRate * gradient;
                    } else if (optimizer == "RMSProp") {
                        cache_rmsprop[l][i][j] = beta_rmsprop * cache_rmsprop[l][i][j] + (1 - beta_rmsprop) * gradient * gradient;
                        layer.weights[i][j] -= learningRate * gradient / (sqrt(cache_rmsprop[l][i][j]) + 1e-8);
                    } else if (optimizer == "Adam") {
                        m_adam[l][i][j] = beta1_adam * m_adam[l][i][j] + (1 - beta1_adam) * gradient;
                        v_adam[l][i][j] = beta2_adam * v_adam[l][i][j] + (1 - beta2_adam) * gradient * gradient;

                        double m_hat = m_adam[l][i][j] / (1 - pow(beta1_adam, t_adam));
                        double v_hat = v_adam[l][i][j] / (1 - pow(beta2_adam, t_adam));

                        layer.weights[i][j] -= learningRate * m_hat / (sqrt(v_hat) + 1e-8);
                    }
                }
                layer.biases[i] -= learningRate * layer.errors[i];
            }
        }
    }
};

#endif
