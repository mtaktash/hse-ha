import numpy as np


class Layer:
    """
    A building block. Each layer is capable of performing two things:

    - Process input to get output:           output = layer.forward(input)

    - Propagate gradients through itself:    grad_input = layer.backward(input, grad_output)

    Some layers also have learnable parameters which they update during layer.backward.
    """

    def __init__(self):
        """
        Here you can initialize layer parameters (if any) and auxiliary stuff.
        """

        raise NotImplementedError("Not implemented in interface")

    def forward(self, input):
        """
        Takes input data of shape [batch, ...], returns output data [batch, ...]
        """

        raise NotImplementedError("Not implemented in interface")

    def backward(self, input, grad_output):
        """
        Performs a backpropagation step through the layer, with respect to the given input. Updates layer parameters and returns gradient for next layer
        Let x be layer weights, output – output of the layer on the given input and grad_output – gradient of layer with respect to output

        To compute loss gradients w.r.t parameters, you need to apply chain rule (backprop):
        (d loss / d x)  = (d loss / d output) * (d output / d x)
        Luckily, you already receive (d loss / d output) as grad_output, so you only need to multiply it by (d output / d x)
        If your layer has parameters (e.g. dense layer), you need to update them here using d loss / d x

        returns (d loss / d input) = (d loss / d output) * (d output / d input)
        """

        raise NotImplementedError("Not implemented in interface")


class ReLU(Layer):
    def __init__(self):
        """
        ReLU layer simply applies elementwise rectified linear unit to all inputs
        This layer does not have any parameters.
        """

        pass

    def forward(self, input):
        """
        Perform ReLU transformation
        input shape: [batch, input_units]
        output shape: [batch, input_units]
        """

        return input * (input > 0)

    def backward(self, input, grad_output):
        """
        Compute gradient of loss w.r.t. ReLU input
        """

        return grad_output * (input > 0)


class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        """
        A dense layer is a layer which performs a learned affine transformation:
        f(x) = Wx + b

        W: matrix of shape [num_inputs, num_outputs]
        b: vector of shape [num_outputs]
        """

        self.learning_rate = learning_rate

        # initialize weights with small random numbers from normal distribution
        self.weights = np.random.normal(scale=np.sqrt(1 / input_units), size=(input_units, output_units))
        self.biases = np.random.normal(scale=np.sqrt(1 / input_units), size=output_units)

    def forward(self, input):
        """
        Perform an affine transformation:
        f(x) = <W*x> + b

        input shape: [batch, input_units]
        output shape: [batch, output units]
        """

        return np.matmul(input, self.weights) + self.biases

    def backward(self, input, grad_output):
        """
        input shape: [batch, input_units]
        grad_output: [batch, output units]

        Returns: grad_input, gradient of output w.r.t input
        """

        grad_loss_input = np.matmul(grad_output, self.weights.T)
        grad_loss_weights = np.matmul(input.T, grad_output)
        grad_loss_biases = np.matmul(np.ones(input.shape[0]), grad_output)

        self.weights -= self.learning_rate * grad_loss_weights
        self.biases -= self.learning_rate * grad_loss_biases

        return grad_loss_input


# Layer ideas from http://cs231n.github.io/assignments2015/assignment2/

class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, learning_rate=0.1):
        """
        A convolutional layer with out_channels kernels of kernel_size.

        in_channels — number of input channels
        out_channels — number of convolutional filters
        kernel_size — tuple of two numbers: k_1 and k_2

        Initialize required weights.
        """

        self.learning_rate = learning_rate
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.h_kernel, self.w_kernel = kernel_size
        h_kernel, w_kernel = kernel_size
        self.weights = np.random.normal(scale=np.sqrt(1 / (in_channels * h_kernel * w_kernel )),
                                        size=(in_channels, out_channels, h_kernel, w_kernel))

    def forward(self, input):
        """
        Perform convolutional transformation:

        input shape: [batch, in_channels, h, w]
        output shape: [batch, out_channels, h_out, w_out]
        """

        batch_size, in_channels, h, w = input.shape
        h_out = h - self.h_kernel + 1
        w_out = w - self.w_kernel + 1

        output = np.zeros((batch_size, self.out_channels, h_out, w_out))

        for n in range(batch_size):
            batch_n = input[n,:,:,:]
            for c in range(self.out_channels):
                for h0 in range(h_out):
                    for w0 in range(w_out):
                        h1, w1 = h0 + self.h_kernel, w0 + self.w_kernel
                        window = batch_n[:,h0:h1,w0:w1]
                        output[n, c, h0, w0] = np.sum(window * self.weights[:,c,:,:])

        return output

    def backward(self, input, grad_output):
        """
        Compute gradients w.r.t input and weights and update weights

        grad_output shape: [batch, out_channels, h_out, w_out]
        """

        batch_size, in_channels, h, w = input.shape
        _, _, h_out, w_out = grad_output.shape

        grad_input = np.zeros_like(input)
        grad_weights = np.zeros_like(self.weights)

        for n in range(batch_size):
            grad_batch_n = grad_input[n,:,:,:]
            batch_n = input[n,:,:,:]

            for c in range(self.out_channels):
                for h0 in range(h_out):
                    for w0 in range(w_out):
                        h1, w1 = h0 + self.h_kernel, w0 + self.w_kernel

                        grad_batch_n[:, h0:h1, w0:w1] += self.weights[:,c,:,:] * grad_output[n,c,h0,w0]
                        grad_weights[:,c,:,:] += batch_n[:, h0:h1, w0:w1] * grad_output[n,c,h0,w0]

            grad_input[n,:,:,:] = grad_batch_n

        self.weights -= self.learning_rate * grad_weights

        return grad_input


class Maxpool2d(Layer):
    def __init__(self, kernel_size):
        """
        A maxpooling layer with kernel of kernel_size.
        This layer donwsamples [kernel_size, kernel_size] to
        1 number which represents maximum.

        Stride description is identical to the convolution
        layer. But default value we use is kernel_size to
        reduce dim by kernel_size times.

        This layer does not have any learnable parameters.
        """

        self.stride = kernel_size
        self.kernel_size = kernel_size

    def forward(self, input):
        """
        Perform maxpooling transformation:

        input shape: [batch, in_channels, h, w]
        output shape: [batch, out_channels, h_out, w_out]
        """

        batch_size, in_channels, h, w = input.shape
        input_reshaped = input.reshape(batch_size, in_channels,
                              h // self.kernel_size, self.kernel_size,
                              w // self.kernel_size, self.kernel_size)

        return np.max(input_reshaped, axis=(3, 5))

    def backward(self, input, grad_output):
        """
        Compute gradient of loss w.r.t. Maxpool2d input
        """

        batch_size, in_channels, h, w = input.shape
        input_reshaped  = input.reshape(batch_size, in_channels,
                              h // self.kernel_size, self.kernel_size,
                              w // self.kernel_size, self.kernel_size)

        output = np.max(input_reshaped, axis=(3, 5))
        output_newaxis = output[:, :, :, np.newaxis, :, np.newaxis]

        mask = (input_reshaped == output_newaxis)

        grad_input_reshaped = np.zeros_like(input_reshaped)
        grad_output_newaxis = grad_output[:, :, :, np.newaxis, :, np.newaxis]

        grad_output_broadcast, _ = np.broadcast_arrays(grad_output_newaxis, grad_input_reshaped)
        grad_input_reshaped[mask] = grad_output_broadcast[mask]
        grad_input_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
        grad_input = grad_input_reshaped.reshape(input.shape)

        return grad_input


class Flatten(Layer):
    def __init__(self):
        """
        This layer does not have any parameters
        """

        pass

    def forward(self, input):
        """
        input shape: [batch_size, channels, feature_nums_h, feature_nums_w]
        output shape: [batch_size, channels * feature_nums_h * feature_nums_w]
        """

        batch_size, channels, feature_nums_h, feature_nums_w = input.shape

        return input.reshape(batch_size, channels * feature_nums_h * feature_nums_w)

    def backward(self, input, grad_output):
        """
        Compute gradient of loss w.r.t. Flatten input
        """

        batch_size, channels, feature_nums_h, feature_nums_w = input.shape

        return grad_output.reshape(batch_size, channels, feature_nums_h, feature_nums_w)


# formulas from https://deepnotes.io/softmax-crossentropy
# one hot encoding https://www.reddit.com/r/MachineLearning/comments/31fk7i/converting_target_indices_to_onehotvector/

def softmax_crossentropy_with_logits(logits, y_true):
    """
    Compute crossentropy from logits and ids of correct answers
    logits shape: [batch_size, num_classes]
    reference_answers: [batch_size]
    output is a number
    """

    exps = np.exp(logits)
    softmax = exps / np.sum(exps, axis=1)[:, np.newaxis]
    y_matrix = np.zeros_like(logits)
    y_matrix[np.arange(logits.shape[0]), y_true] = 1

    return -np.sum(np.log(softmax) * y_matrix) / y_true.shape[0]


def grad_softmax_crossentropy_with_logits(logits, y_true):
    """
    Compute crossentropy gradient from logits and ids of correct answers
    Output should be divided by batch_size, so any layer update can be simply computed as sum of object updates.
    logits shape: [batch_size, num_classes]
    reference_answers: [batch_size]
    """

    exps = np.exp(logits)
    softmax = exps / np.sum(exps, axis=1)[:, np.newaxis]
    y_matrix = np.zeros_like(logits)
    y_matrix[np.arange(logits.shape[0]), y_true] = 1

    return (softmax - y_matrix) / y_true.shape[0]

