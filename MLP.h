#include <vector>
#include <random>
#include <fstream> 
#include "Layer.h"
#include "softmax.h"
#include "ActivationFunction.h"

using namespace std;

enum class OptimizerType {
    SGD,
    Adam,
    RMSProp
};


#ifndef MULTILAYERPERCEPTRON_H
#define MULTILAYERPERCEPTRON_H

class MultiLayerPerceptron {
public:
    vector<Layer> layers;
    double learningRate = 0.1;
    size_t numLayers = 0;
    OptimizerType optimizer = OptimizerType::SGD;

    MultiLayerPerceptron(
        const vector<int>& layerSizes,
        const vector<ActivationFunction>& activations,
        double learningRate = 0.1,
        OptimizerType optimizer = OptimizerType::SGD
    ) : learningRate(learningRate), optimizer(optimizer)
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

    void backward_from_softmax(const vector<double>& input, const vector<double>& target, const vector<double>& probs) {
       this->forward(input); // solo si aún no has guardado las salidas
        vector<double> output_error(probs.size());

        // Aquí aplicas la derivada simplificada
        for (size_t i = 0; i < probs.size(); ++i) {
            output_error[i] = probs[i] - target[i];
        }

        layers.back().errors = output_error;

        // Backpropagación desde la penúltima capa
        for (int l = layers.size() - 2; l >= 0; --l) {
            Layer& current = layers[l];
            Layer& next = layers[l + 1];

            for (int i = 0; i < current.numOutputs; ++i) {
                double error = 0.0;
                for (int j = 0; j < next.numOutputs; ++j) {
                    error += next.weights[j][i] * next.errors[j];
                }
                current.errors[i] = error * current.activation.derivative(current.outputs[i]);
            }
        }

        // Actualizar pesos y sesgos
        updateWeights(input);
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
    static int t = 0;
    t++;

    double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
    double rho = 0.9;  // Para RMSProp

    for (int l = 0; l < numLayers; ++l) {
        Layer& layer = layers[l];
        const vector<double>& in = l == 0 ? input : layers[l - 1].outputs;
for (int i = 0; i < layer.numOutputs; ++i) {
    for (int j = 0; j < layer.numInputs; ++j) {
        double grad = layer.errors[i] * in[j];

        switch (optimizer) {
            case OptimizerType::SGD: {
                layer.weights[i][j] -= learningRate * grad;
                break;
            }

            case OptimizerType::Adam: {
                layer.m_weights[i][j] = beta1 * layer.m_weights[i][j] + (1 - beta1) * grad;
                layer.v_weights[i][j] = beta2 * layer.v_weights[i][j] + (1 - beta2) * grad * grad;

                double m_hat = layer.m_weights[i][j] / (1 - pow(beta1, t));
                double v_hat = layer.v_weights[i][j] / (1 - pow(beta2, t));
                layer.weights[i][j] -= learningRate * m_hat / (sqrt(v_hat) + epsilon);
                break;
            }

            case OptimizerType::RMSProp: {
                layer.v_weights[i][j] = rho * layer.v_weights[i][j] + (1 - rho) * grad * grad;
                layer.weights[i][j] -= learningRate * grad / (sqrt(layer.v_weights[i][j]) + epsilon);
                break;
            }
        }
    }

    double grad_b = layer.errors[i];

    switch (optimizer) {
        case OptimizerType::SGD: {
            layer.biases[i] -= learningRate * grad_b;
            break;
        }

        case OptimizerType::Adam: {
            layer.m_biases[i] = beta1 * layer.m_biases[i] + (1 - beta1) * grad_b;
            layer.v_biases[i] = beta2 * layer.v_biases[i] + (1 - beta2) * grad_b * grad_b;

            double m_hat_b = layer.m_biases[i] / (1 - pow(beta1, t));
            double v_hat_b = layer.v_biases[i] / (1 - pow(beta2, t));
            layer.biases[i] -= learningRate * m_hat_b / (sqrt(v_hat_b) + epsilon);
            break;
        }

        case OptimizerType::RMSProp: {
            layer.v_biases[i] = rho * layer.v_biases[i] + (1 - rho) * grad_b * grad_b;
            layer.biases[i] -= learningRate * grad_b / (sqrt(layer.v_biases[i]) + epsilon);
            break;
        }
    }
}

    }
}



};

#endif
