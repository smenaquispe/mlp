// Optimizer.h

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
using namespace std;

class Optimizer
{
public:
    virtual void update(
        vector<vector<double>> &weights,
        vector<double> &biases,
        const vector<double> &inputs,
        const vector<double> &errors) = 0;

    virtual ~Optimizer() {}
};

#endif
