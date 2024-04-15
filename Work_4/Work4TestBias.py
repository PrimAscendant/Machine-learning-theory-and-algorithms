import numpy as np

class Neuron:
    def __init__(self, initial_bias=0.1):
        self.initialize_weights(initial_bias)

    def initialize_weights(self, bias):
        self.weights = np.random.rand(2)
        self.bias = bias

    def activation(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, inputs):
        return self.activation(np.dot(inputs, self.weights) + self.bias)
    
    def train(self, training_data, epochs=1000, learning_rate=0.1):
        history = []
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in training_data:
                prediction = self.predict(inputs)
                error = target - prediction
                total_error += abs(error)
                self.weights += learning_rate * error * np.array(inputs)
                self.bias += learning_rate * error
            history.append((self.weights.copy(), self.bias))
            if total_error == 0:
                break
        return epoch + 1, history  # Number of epochs and history

def evaluate_bias_effect(operation_data, bias_values):
    results = {}
    for bias in bias_values:
        neuron = Neuron(initial_bias=bias)
        epochs_needed, history = neuron.train(operation_data)
        results[bias] = epochs_needed
    return results

# Define your logical operation data here
operation_data = [(np.array([0, 0]), 0), (np.array([0, 1]), 1), (np.array([1, 0]), 1), (np.array([1, 1]), 1)]  # Example for OR

# Test different bias values
bias_values = np.linspace(0, 1, 20)  # Test 20 different values from 0 to 1
results = evaluate_bias_effect(operation_data, bias_values)
for bias, epochs in results.items():
    print(f"Bias: {bias:.2f}, Epochs: {epochs}")
