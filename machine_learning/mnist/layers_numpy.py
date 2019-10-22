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


# Fancy indexing from http://cs231n.github.io/assignments2015/assignment2/

def get_im2col_indices(input_shape, h_kernel, w_kernel):

    batch, in_channel, h, w = input_shape

    h_out = h - h_kernel + 1
    w_out = w - w_kernel + 1

    i0 = np.repeat(np.arange(h_kernel), w_kernel)
    i0 = np.tile(i0, in_channel)
    i_bias = np.repeat(np.arange(h_out), w_out)

    j0 = np.tile(np.arange(w_kernel), h_kernel * in_channel)
    j_bias = np.tile(np.arange(w_out), h_out)

    i = i0.reshape(-1, 1) + i_bias.reshape(1, -1)
    j = j0.reshape(-1, 1) + j_bias.reshape(1, -1)
    k = np.repeat(np.arange(in_channel), h_kernel * w_kernel).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(input, h_kernel, w_kernel):

    k, i, j = get_im2col_indices(input.shape, h_kernel, w_kernel)

    cols = input[:, k, i, j]

    in_channel = input.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(h_kernel * w_kernel * in_channel, -1)

    return cols


def col2im_indices(cols, input_shape, h_kernel=3, w_kernel=3):

    batch, in_channel, h, w = input_shape
    output = np.zeros((batch, in_channel, h, w), dtype=cols.dtype)

    k, i, j = get_im2col_indices(input_shape, h_kernel, w_kernel)

    cols_reshaped = cols.reshape(in_channel * h_kernel * w_kernel, -1, batch)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    output[slice(None), k, i, j] = cols_reshaped

    return output

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
                                        size=(out_channels, in_channels, h_kernel, w_kernel))


    def forward(self, input):
        """
        Perform convolutional transformation:

        input shape: [batch, in_channels, h, w]
        output shape: [batch, out_channels, h_out, w_out]
        """

        batch_size, in_channels, h, w = input.shape
        h_out = h - self.h_kernel + 1
        w_out = w - self.w_kernel + 1

        input_cols = im2col_indices(input, self.h_kernel, self.w_kernel)
        weights_cols = self.weights.reshape(self.out_channels, -1)

        output = np.matmul(weights_cols, input_cols)
        output = output.reshape(self.out_channels, h_out, w_out, batch_size)
        output = output.transpose(3, 0, 1, 2)

        return output

    def backward(self, input, grad_output):
        """
        Compute gradients w.r.t input and weights and update weights

        grad_output shape: [batch, out_channels, h_out, w_out]
        """

        grad_output_reshaped = grad_output.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        input_cols = im2col_indices(input, self.h_kernel, self.w_kernel)
        grad_weights = np.matmul(grad_output_reshaped, input_cols.T)
        grad_weights = grad_weights.reshape(self.weights.shape)

        weights_reshaped = self.weights.reshape(self.out_channels, -1)
        grad_input_cols = np.matmul(weights_reshaped.T, grad_output_reshaped)
        grad_input = col2im_indices(grad_input_cols, input.shape, self.h_kernel, self.w_kernel)

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

