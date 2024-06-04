import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_weights():
    weights = np.random.randn(1, 1)
    bias = np.random.randn(1)
    return weights, bias

def forward_propagation(X, weights, bias):
    weighted_sum = np.dot(X, weights) + bias
    output = sigmoid(weighted_sum)
    return output

def backward_propagation(X, y_true, output, weights, bias, learning_rate):
    error = y_true - output
    d_output = error * sigmoid_derivative(output)
    d_weights = np.dot(X.T, d_output)
    d_bias = np.sum(d_output)

    weights += learning_rate * d_weights
    bias += learning_rate * d_bias

    return weights, bias

def train(X, y_true, learning_rate, epochs):
    weights, bias = initialize_weights()

    for epoch in range(epochs):
        output = forward_propagation(X, weights, bias)
        weights, bias = backward_propagation(X, y_true, output, weights, bias, learning_rate)

        if epoch % 100 == 0:
            loss = np.mean(np.square(y_true - output))
            print(f'Epoch: {epoch}, Loss: {loss:.4f}')

    return weights, bias

def predict(X, weights, bias):
    return forward_propagation(X, weights, bias)

data = pd.read_csv('data.csv')
height = np.array(data.iloc[:, 0].values).reshape(-1, 1)
weight = np.array(data.iloc[:, 1].values).reshape(-1, 1)

# Normalize data
normalized_height = height * (1 / np.max(height))
normalized_weight = weight * (1 / np.max(weight))

# Train the neural network
trained_weights, trained_bias = train(normalized_height, normalized_weight, learning_rate=0.1, epochs=10000)

# Predict weight for a new height value
row = 7
n_height = np.array(data.iloc[row, 0])
n_weight = np.array(data.iloc[row, 1])
normalized_n_height = n_height * (1 / np.max(height))
predicted_weight = predict(normalized_n_height, trained_weights, trained_bias) * np.max(weight)
print(f'Predicted weight for height {n_height} cm: {predicted_weight[0][0]:.2f} kg')
print(f'True weight for height {n_height} cm: {n_weight:.2f} kg')
