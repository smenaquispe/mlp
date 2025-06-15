// Optimizer.h

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
using namespace std;

class Optimizer
{
public:
    virtual void update(
        std::vector<std::vector<double>> &weights,
        std::vector<double> &biases,
        const std::vector<double> &inputs,
        const std::vector<double> &errors) = 0;

    virtual void update_bias(
        std::vector<double> &biases,
        const std::vector<double> &errors,
        int layerIndex) {} // Puedes dejarlo vacío por defecto

    virtual ~Optimizer() {}
};

#endif
