// SGDOptimizer.h

#ifndef SGDOPTIMIZER_H
#define SGDOPTIMIZER_H

#include "Optimizer.h"
#include "Mat.h"

class SGDOptimizer : public Optimizer
{
public:
    double learningRate;

    SGDOptimizer(double lr) : learningRate(lr) {}

    void update(
        vector<vector<double>> &weights,
        vector<double> &biases,
        const vector<double> &inputs,
        const vector<double> &errors) override
    {
        int numOutputs = errors.size();
        int numInputs = inputs.size();

        for (int i = 0; i < numOutputs; ++i)
        {
            for (int j = 0; j < numInputs; ++j)
            {
                weights[i][j] -= learningRate * errors[i] * inputs[j];
            }
            biases[i] -= learningRate * errors[i];
        }
    }
};

#endif
