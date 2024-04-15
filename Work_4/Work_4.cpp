// CLEAR   g++ work Work_4.cpp -std=c++11

#include <vector>
#include <iostream>
#include <random>
#include <numeric> // For std::accumulate 

class Neuron {
public:
    std::vector<double> weights;
    double bias;

    Neuron() {
        std::random_device rd;  
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> distrib(0.0, 1.0); 

        weights = {distrib(gen), distrib(gen)};
        bias = distrib(gen);
    }

    double activation(double x) {
        return x >= 0 ? 1 : 0;
    }

    double predict(const std::vector<double>& inputs) {
        return activation(dot_product(inputs, weights) + bias);
    }

    double dot_product(const std::vector<double>& v1, const std::vector<double>& v2) {
        double result = 0.0;
        for (size_t i = 0; i < v1.size(); i++) {
            result += v1[i] * v2[i];
        }
        return result;
    }

    void train(const std::vector<std::pair<std::vector<double>, double>>& training_data, int epochs = 1000, double learning_rate = 0.1) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double total_error = 0;
            for (const auto& data : training_data) {
                double prediction = predict(data.first);
                double error = data.second - prediction;
                total_error += std::abs(error);
                for (size_t i = 0; i < weights.size(); i++) {
                    weights[i] += learning_rate * error * data.first[i];
                }
                bias += learning_rate * error;
            }
            if (total_error == 0) {
                break;
            }
        }
    }
};

int main() {
    Neuron n;
    std::vector<std::pair<std::vector<double>, double>> training_data = {{{0, 0}, 0}, {{0, 1}, 1}, {{1, 0}, 1}, {{1, 1}, 1}};
    n.train(training_data);

    std::cout << "Trained weights: ";
    for (auto w : n.weights) {
        std::cout << w << " ";
    }
    std::cout << "\nBias: " << n.bias << std::endl;

    return 0;
}
