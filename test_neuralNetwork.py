from unittest import TestCase
import numpy as np

from NeuralNetwork import NeuralNetwork


class TestNeuralNetwork(TestCase):

    def test_neural_net(self):
        x = np.array([[0, 0, 1],
                      [0, 1, 1],
                      [1, 0, 1],
                      [1, 1, 1]])
        y = np.array([[0], [1], [1], [0]])
        nn = NeuralNetwork(x, y)
        for _ in range(100):
            nn.feedforward()
            nn.backprop()
        print(nn.output)
        self.assertTrue(np.array_equal(y, np.round(nn.output)))
