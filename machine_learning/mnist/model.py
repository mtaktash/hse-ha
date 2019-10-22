import numpy as np
from layers_numpy import *


class NeuralNetwork:
    def __init__(self, layers):
        """
        layers — list of Layer objects
        """

        self.layers = layers

    def forward(self, X):
        """
        Compute activations of all network layers by applying them sequentially.
        Return a list of activations for each layer.
        Make sure last activation corresponds to network logits.
        """

        activations = []
        input = X

        for layer in self.layers:
            input = layer.forward(input)
            activations.append(input)

        assert len(activations) == len(self.layers)
        return activations

    def predict(self, X):
        """
        Use network to predict the most likely class for each sample.
        """

        layer_activations = self.forward(X)
        logits = layer_activations[-1]
        exp_logits = np.exp(logits)
        softmax = exp_logits / np.sum(exp_logits, axis=1)[:, np.newaxis]

        return np.argmax(softmax, axis=1)

    def backward(self, X, y):
        """
        Train your network on a given batch of X and y.
        You first need to run forward to get all layer activations.
        Then you can run layer.backward going from last to first layer.

        After you called backward for all layers, all Dense layers have already made one gradient step.
        """

        # Get the layer activations
        layer_activations = self.forward(X)
        layer_inputs = [X] + layer_activations  # layer_input[i] is an input for network[i]
        logits = layer_activations[-1]

        # Compute the loss and the initial gradient
        loss = softmax_crossentropy_with_logits(logits, y)
        loss_grad = grad_softmax_crossentropy_with_logits(logits, y)

        # propagate gradients through network layers using .backward
        # hint: start from last layer and move to earlier layers
        layers_with_inputs = list(zip(self.layers, layer_inputs))
        for layer, layer_input in layers_with_inputs[::-1]:
            loss_grad = layer.backward(layer_input, loss_grad)

        return np.mean(loss)

