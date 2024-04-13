import numpy as np

class Neuron:
    def __init__(self):
        self.initialize_weights()

    def initialize_weights(self):
        self.weights = np.random.rand(2)
        self.bias = np.random.rand(1)[0]

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
        return epoch + 1, history  # Кількість епох і історія ваг і порогу

def evaluate_operations(operations, trials=5):
    results = {}
    for name, data in operations.items():
        epochs_list = []
        for _ in range(trials):
            neuron = Neuron()
            epochs_needed, history = neuron.train(data)
            epochs_list.append(epochs_needed)
        results[name] = {
            'average_epochs': np.mean(epochs_list),
            'epochs_list': epochs_list
        }
    return results

# Перевірка на здатність нейрона навчитися різним операціям
logical_operations = {
    'OR': [(np.array([0, 0]), 0), (np.array([0, 1]), 1), (np.array([1, 0]), 1), (np.array([1, 1]), 1)],
    'AND': [(np.array([0, 0]), 0), (np.array([0, 1]), 0), (np.array([1, 0]), 0), (np.array([1, 1]), 1)],
    'NAND': [(np.array([0, 0]), 1), (np.array([0, 1]), 1), (np.array([1, 0]), 1), (np.array([1, 1]), 0)],
    'NOR': [(np.array([0, 0]), 1), (np.array([0, 1]), 0), (np.array([1, 0]), 0), (np.array([1, 1]), 0)],
    'XOR': [(np.array([0, 0]), 0), (np.array([0, 1]), 1), (np.array([1, 0]), 1), (np.array([1, 1]), 0)]
}

results = evaluate_operations(logical_operations)
for operation, result in results.items():
    print(f"Логічна операція: {operation}")
    print(f"Середня кількість епох: {result['average_epochs']}")
    print(f"Список кількості епох по спробах: {result['epochs_list']}")
