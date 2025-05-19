#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <functional> 

using namespace std;

class Perceptron {
private:
    std::vector<float> weights;
    std::function<int(float)> activationFunction;
    float learningRate;
    float bias;

public:
    Perceptron(
            int inputSize,  // numero de inputs
            std::function<int(float)> activation, // funcion de activacion 
            float lr = 0.1,  // tasa de aprendizaje
            float b = 1.0f // el bias
        )
        :   learningRate(lr), 
            activationFunction(activation), 
            bias(b) 
        {
            // valores random para los pesos
            weights.resize(inputSize);
            srand(time(0));
            for (auto& w : weights) w = (float(rand()) / RAND_MAX) * 2 - 1;
        }

    float test(const std::vector<int>& inputs) {
        // calcular la salida con la funcion se sumatoria
        float sum = bias;
        for (size_t i = 0; i < inputs.size(); ++i)
            sum += inputs[i] * weights[i];
        return activationFunction(sum);
    }

    void train(
        const std::vector<std::vector<int>>& inputs, // las entradas
        const std::vector<int>& labels, // los labels son los valores esperados
        int epochs
    ) 
    {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (size_t i = 0; i < inputs.size(); i++) {
                int output = test(inputs[i]);
                int error = labels[i] - output; // el error es la diferencia de la salida esperada y la salida real
                
                // vamos actualizando los nuevos pesos
                for (size_t j = 0; j < weights.size(); j++)
                    weights[j] += learningRate * error * inputs[i][j];
                
                bias += learningRate * error;


                cout << "Epoch: " << epoch << ", Input: " << inputs[i][0] << ", " << inputs[i][1]
                     << ", Output: " << output << ", Error: " << error
                     << ", Weights: ";
                
                for (const auto& w : weights) cout << w << " ";
                
                cout << ", Bias: " << bias << endl;

            }
        }
    }

};


int main() {

    // la funcion de activacion es la funcion step
    auto activation_function = [](float x) {
        return x >= 0 ? 1 : 0;
    };
    
    Perceptron p(2, activation_function, 0.4);

    // las entradas son los valores logicos las 4 combinaciones
    // 0 0
    // 0 1
    // 1 0
    // 1 1
    std::vector<std::vector<int>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };

    // valores esperados para la funcion AND
    // esta en el mismo orden que las entradas
    std::vector<int> labels_and = {0, 0, 0, 1};

    // valores esperados para la funcion OR
    std::vector<int> labels_or = {0, 1, 1, 1};

    p.train(inputs, labels_and, 10);

    std::cout << "AND results:\n";
    for (const auto& input : inputs)
        std::cout << input[0] << " AND " << input[1] << " = " << p.test(input) << "\n";

    return 0;
}