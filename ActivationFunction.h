// ActivationFunction.h

#pragma once
#include <cmath>
#include <functional>

using namespace std;

#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H

class ActivationFunction {
    public:
        function<double(double)> func;
        function<double(double)> derivative;
    
        ActivationFunction(function<double(double)> f, function<double(double)> df) 
            : func(f), 
            derivative(df) 
        {}
        
};

#endif


