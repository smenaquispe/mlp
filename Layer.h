// Layer.h

#include <vector>
#include <random>
#include "ActivationFunction.h"

using namespace std;

#ifndef LAYER_H
#define LAYER_H

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

        uniform_int_distribution <> getRandomDistribution(int min, int max) {
            return uniform_int_distribution<>(min, max);
        }


        void setRandomWeights() {
            random_device rd;
            mt19937 gen(rd());
            double stddev = sqrt(2.0 / numInputs);
            normal_distribution<double> dis(0.0, stddev);

            for (int i = 0; i < numOutputs; ++i) {
                for (int j = 0; j < numInputs; ++j) {
                    weights[i][j] = dis(gen);
                }
                biases[i] = 0.01;
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