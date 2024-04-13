#include <vector>
#include <iostream>
#include <random>
#include <map>
#include <numeric> // For std::accumulate

class Neuron {
public:
    double initialize_weights;
    double bias;

    Neuron() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> distrib(0.0, 1.0);
        initialize_weights = distrib(gen); // Generate a random number
        bias = distrib(gen); // Generate a random number for bias
    }

    double dot_predict(const std::vector<double>& v1, const std::vector<double>& v2) {
        double result = 0.0;
        for (size_t i = 0; i < v1.size(); i++) {
            result += v1[i] * v2[i];
        }
        return result;
    }

    std::vector<double> train(const std::vector<double>& v1, const std::vector<double>& v2, std::vector<double>& history, double epochs, double learning_rate, double predict) {
        double error = 0;
        for (int t = 0; t < epochs; t++) {
            error = v2[t] - predict; // Assuming v2 has enough elements
            initialize_weights += learning_rate * error * v1[t]; // Assuming v1 has enough elements
            history.push_back(initialize_weights);
            bias += learning_rate * error;
            history.push_back(bias);
        }
        return history;
    }
};

int main() {
    Neuron n;
    std::vector<double> vector1 = {1.0, 2.0, 3.0};
    std::vector<double> vector2 = {4.0, 5.0, 6.0};
    std::vector<double> history;

    double predict = n.dot_predict(vector1, vector2);
    std::cout << "Initial Prediction: " << predict << std::endl;

    history = n.train(vector1, vector2, history, 10, 0.1, predict);
    std::cout << "Training complete." << std::endl;

    return 0;
}
