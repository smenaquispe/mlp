// softwax.h

#ifndef SOFTMAX_H
#define SOFTMAX_H
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;

vector<double> softmax(const vector<double>& input) {
    vector<double> output;
    double max_logit = *max_element(input.begin(), input.end());

    double sum = 0.0;
    for (double logit : input) {
        sum += exp(logit - max_logit);
    }

    for (double logit : input) {
        output.push_back(exp(logit - max_logit) / sum);
    }

    return output;
}

#endif