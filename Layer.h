#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <random>
#include "ActivationFunction.h"

using namespace std;

class Layer {
public:
    int numInputs;
    int numOutputs;

    vector<vector<double>> weights;
    vector<double> biases;
    vector<double> outputs;
    vector<double> inputs;
    vector<double> errors;
    ActivationFunction activation;

    // === Parámetros para Adam y RMSProp ===
    vector<vector<double>> m_weights; // Momento 1 para pesos
    vector<vector<double>> v_weights; // Momento 2 para pesos
    vector<double> m_biases;          // Momento 1 para bias
    vector<double> v_biases;          // Momento 2 para bias

    Layer(int numInputs, int numOutputs, ActivationFunction activation)
        : numInputs(numInputs), 
          numOutputs(numOutputs), 
          activation(activation) 
    {
        weights.resize(numOutputs, vector<double>(numInputs));
        biases.resize(numOutputs);
        outputs.resize(numOutputs);
        inputs.resize(numInputs);
        errors.resize(numOutputs);

        // Inicializar momentos para optimizadores
        m_weights.resize(numOutputs, vector<double>(numInputs, 0.0));
        v_weights.resize(numOutputs, vector<double>(numInputs, 0.0));
        m_biases.resize(numOutputs, 0.0);
        v_biases.resize(numOutputs, 0.0);

        this->setRandomWeights();
        this->setRandomBiases();
    }

    vector<double> forward(const vector<double>& inputs) {
        this->inputs = inputs;
        for (int i = 0; i < numOutputs; ++i) {
            double sum = biases[i];
            for (int j = 0; j < numInputs; ++j)
                sum += weights[i][j] * inputs[j];
            outputs[i] = activation.func(sum);
        }
        return outputs;
    }

private:
    mt19937 getRandomGenerator() {
        random_device rd;
        return mt19937(rd());
    }

    uniform_real_distribution<> getRandomDistribution(double min, double max) {
        return uniform_real_distribution<>(min, max);
    }

    void setRandomWeights() {
        mt19937 gen = getRandomGenerator();
        auto dis = getRandomDistribution(-1.0, 1.0);
        for (int i = 0; i < numOutputs; ++i) {
            for (int j = 0; j < numInputs; ++j) {
                weights[i][j] = dis(gen);
            }
        }
    }

    void setRandomBiases() {
        mt19937 gen = getRandomGenerator();
        auto dis = getRandomDistribution(-1.0, 1.0);
        for (int i = 0; i < numOutputs; ++i) {
            biases[i] = dis(gen);
        }
    }
};

#endif
