// Layer.h

#ifndef LAYER_H
#define LAYER_H

#include "Mat.h"
#include "ActivationFunction.h"

class Layer
{
public:
    int numInputs;
    int numOutputs;

    Mat weights;
    vector<double> biases;
    vector<double> outputs;
    vector<double> inputs;
    vector<double> errors;

    ActivationFunction activation;

    Layer(int numInputs, int numOutputs, ActivationFunction activation)
        : numInputs(numInputs),
          numOutputs(numOutputs),
          activation(activation),
          weights(numOutputs, numInputs, true),
          biases(numOutputs, 0.01),
          outputs(numOutputs),
          inputs(numInputs),
          errors(numOutputs) {}

    vector<double> forward(const vector<double> &input, double dropoutRate = 0.0, bool training = false)
    {
        inputs = input;
        outputs = weights.dot(input);

        for (int i = 0; i < numOutputs; ++i)
        {
            outputs[i] = activation.func(outputs[i] + biases[i]);

            if (training && dropoutRate > 0.0)
            {
                double randVal = static_cast<double>(rand()) / RAND_MAX;
                if (randVal < dropoutRate)
                {
                    outputs[i] = 0.0;
                }
                else
                {
                    outputs[i] /= (1.0 - dropoutRate);
                }
            }
        }

        return outputs;
    }
};

#endif // LAYER_H