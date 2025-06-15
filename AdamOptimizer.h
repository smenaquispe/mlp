#ifndef ADAMOPTIMIZER_H
#define ADAMOPTIMIZER_H

#include "Optimizer.h"
#include <vector>
#include <cmath>

class AdamOptimizer : public Optimizer
{
private:
    double learningRate;
    double beta1;
    double beta2;
    double epsilon;
    double weightDecay;

    int t;

    vector<vector<vector<double>>> m_w, v_w;
    vector<vector<double>> m_b, v_b;

public:
    AdamOptimizer(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8, double wd = 0.01)
        : learningRate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0), weightDecay(wd) {}

    void update(
        vector<vector<double>> &weights,
        vector<double> &biases,
        const vector<double> &inputs,
        const vector<double> &errors) override
    {
        int numOutputs = errors.size();
        int numInputs = inputs.size();

        if (m_w.empty())
        {
            m_w = vector<vector<vector<double>>>(1, vector<vector<double>>(numOutputs, vector<double>(numInputs, 0.0)));
            v_w = m_w;
            m_b = vector<vector<double>>(1, vector<double>(numOutputs, 0.0));
            v_b = m_b;
        }

        t++;

        for (int i = 0; i < numOutputs; ++i)
        {
            for (int j = 0; j < numInputs; ++j)
            {
                // double grad = errors[i] * inputs[j];
                double grad = errors[i] * inputs[j] + weightDecay * weights[i][j];

                m_w[0][i][j] = beta1 * m_w[0][i][j] + (1 - beta1) * grad;
                v_w[0][i][j] = beta2 * v_w[0][i][j] + (1 - beta2) * grad * grad;

                double m_hat = m_w[0][i][j] / (1 - pow(beta1, t));
                double v_hat = v_w[0][i][j] / (1 - pow(beta2, t));

                weights[i][j] -= learningRate * m_hat / (sqrt(v_hat) + epsilon);
            }

            double grad_b = errors[i];

            m_b[0][i] = beta1 * m_b[0][i] + (1 - beta1) * grad_b;
            v_b[0][i] = beta2 * v_b[0][i] + (1 - beta2) * grad_b * grad_b;

            double m_hat_b = m_b[0][i] / (1 - pow(beta1, t));
            double v_hat_b = v_b[0][i] / (1 - pow(beta2, t));

            biases[i] -= learningRate * m_hat_b / (sqrt(v_hat_b) + epsilon);
        }
    }
};

#endif
