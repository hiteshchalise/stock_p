import numpy as np


class NeuralNetwork:
    accuracy = 0

    # temp = 0

    def __init__(self, x, y):
        self.input = x
        self.weights_1 = np.random.rand(self.input.shape[1], 5)
        self.weights_2 = np.random.rand(5, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        # layer1 is hidden layer
        self.z1 = np.dot(self.input, self.weights_1)
        # if NeuralNetwork.temp == 0:
        #     print(self.z1)
        self.layer1 = sigmoid(self.z1)
        # if NeuralNetwork.temp == 0:
        #     print(self.layer1)
        # NeuralNetwork.temp = 1
        self.output = sigmoid(np.dot(self.layer1, self.weights_2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights_2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights_1 = np.dot(self.input.T,
                             (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                     self.weights_2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights_1 += d_weights_1
        self.weights_2 += d_weights_2

    def evaluate(self, testing_x, testing_y):
        self.input = testing_x
        self.y = testing_y
        self.feedforward()
        for index in range(self.y.size):
            if int(self.output[index]) == self.y[index]:
                NeuralNetwork.accuracy += 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(sigmoid_output):
    return sigmoid_output * (1 - sigmoid_output)
