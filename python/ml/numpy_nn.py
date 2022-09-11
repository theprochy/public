import numpy as np
import matplotlib.pyplot as plt
import time

# layer interface - do not modify
class Layer:
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, dL_wrt_output):
        raise NotImplementedError()

    def params(self):
        return {}

    def grads(self):
        return {}

# loss interface - do not modify
class Loss:
    def forward(self, x, y):
        raise NotImplementedError()

    def backward(self):
        raise NotImplementedError()


class Linear(Layer):
    """
    Implements the fully-connected layer with bias, f(x) = Wx + b.
    ("Linear" is another name for "FullyConnected", at least in PyTorch)
    """

    def __init__(self, input_dimension, output_dimension):
        """
        :param input_dimension:   number of inputs (scalar)
        :param output_dimension:  number of outputs (scalar)
        """
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        # random weights and bias initialization - do not change.
        self.W = np.random.randn(input_dimension, output_dimension) / 100
        self.b = np.random.randn(1, output_dimension) / 100

        self.dL_wrt_W = None
        self.dL_wrt_b = None

        self.x = None

    def forward(self, x):
        """
        :param x:  layer input - np.array (batch_sz, input_dimension)
        :return:   layer output - np.array (batch_sz, output_dimension)
        """
        self.x = x.copy()
        return x @ self.W + self.b

    def backward(self, dL_wrt_output):
        """
        :param dL_wrt_output:  gradient of loss wrt layer output - np.array (batch_sz, output_dimension)
        :return:               gradient of loss wrt layer input - np.array (batch_sz, input_dimension)

        note: also compute and store the gradients wrt the layer parameters
               dL_wrt_W - np array (batch_sz, input_dimension, output_dimension)
               dL_wrt_b - np array (batch_sz, 1, output_dimension)
        """
        dL_wrt_x = dL_wrt_output @ self.W.T
        self.dL_wrt_W = self.x[:, :, None] * dL_wrt_output[:, None]
        self.dL_wrt_b = dL_wrt_output[:, None]
        return dL_wrt_x

    def params(self):
        return {'W': self.W, 'b': self.b}

    def grads(self):
        return {'W': self.dL_wrt_W, 'b': self.dL_wrt_b}

class ReLU(Layer):
    """
    ReLU or REctified Linear Unit is a simple yet powerful activation that computes f(x) = max(0, x).
    """

    def __init__(self):
        # need to store forward pass inputs for backpropagation
        self.x = None

    def forward(self, x):
        """
        :param x:  arbitrary shaped np array
        :return:   ReLU activation (same shape as x)
        """
        self.x = x.copy()
        x[x < 0] = 0
        return x

    def backward(self, dL_wrt_output):
        """
        :param dL_wrt_output: gradient of loss wrt layer output (same shape as forward x)
        :return:              gradient of loss wrt layer input (same shape as forward x)
        """
        dL_wrt_x = dL_wrt_output
        dL_wrt_x[self.x < 0] = 0
        return dL_wrt_x

class Sigmoid(Layer):
    """
    Implements a logistic sigmoid activation layer f(x) = 1/(1 + e^-x)
    """

    def __init__(self):
        self.a = None

    def forward(self, x):
        """
        :param x:  arbitrary shaped np array
        :return:   Logistic sigmoid activation (same shape as x)
        """
        self.a = 1 / (1 + np.exp(-x))
        return self.a.copy()

    def backward(self, dL_wrt_output):
        """
        :param dL_wrt_output: gradient of loss wrt layer output (same shape as forward x)
        :return:              gradient of loss wrt layer input (same shape as forward x)
        """
        return self.a * (1 - self.a) * dL_wrt_output


class SE(Loss):
    """
    Implements a mean-squared-error between inputs.
    """
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        """
        :param x:  np array, arbitrary shape
        :param y:  np array, same shape as x
        :return:   squared-error - np array, same shape as x
        """
        self.x = x.copy()
        self.y = y.copy()
        return np.square(x - y)

    def backward(self):
        """
        :return:  gradient wrt the first layer input (x) - np array, same shape as x

        Note: The second input will be the ground truth label, which is fixed.
              No need to propagate gradient there
        """
        return 2 * self.x - 2 * self.y


# Visualization.  You do not have to modify this.
def show_classification(test_images, labels, letters):
    """
    show_classification(test_images, labels, letters)

    create montages of images according to estimated labels

    :param test_images:     np array (h, w, n)
    :param labels:          labels for input images np array (n,)
    :param letters:         string with letters, e.g. 'CN'
    """

    def montage(images, colormap='gray'):
        """
        Show images in grid.

        :param images:      np array (h, w, n)
        :param colormap:    numpy colormap
        """
        h, w, count = np.shape(images)
        h_sq = np.int(np.ceil(np.sqrt(count)))
        w_sq = h_sq
        im_matrix = np.zeros((h_sq * h, w_sq * w))

        image_id = 0
        for j in range(h_sq):
            for k in range(w_sq):
                if image_id >= count:
                    break
                slice_w = j * h
                slice_h = k * w
                im_matrix[slice_h:slice_h + w, slice_w:slice_w + h] = images[:, :, image_id]
                image_id += 1
        plt.imshow(im_matrix, cmap=colormap)
        plt.axis('off')
        return im_matrix

    for i in range(len(letters)):
        imgs = test_images[:, :, labels == i]
        subfig = plt.subplot(1, len(letters), i + 1)
        montage(imgs)
        plt.title(letters[i])

def load_data(path, class_a, class_b):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        x_train = x_train.reshape(-1, 28*28)
        x_test = x_test.reshape(-1, 28*28)

        X_trn = x_train[np.logical_or(y_train == class_a,
                                      y_train == class_b), :]
        y_trn = y_train[np.logical_or(y_train == class_a,
                                      y_train == class_b)]
        y_trn = y_trn.reshape(-1, 1)
        y_trn[y_trn == class_a] = 0
        y_trn[y_trn == class_b] = 1

        y_tst = y_test[np.logical_or(y_test == class_a,
                                    y_test == class_b)]
        y_tst = y_tst.reshape(-1, 1)
        y_tst[y_tst == class_a] = 0
        y_tst[y_tst == class_b] = 1

        # 0-1 normalize the images
        trn_mean = np.mean(X_trn)
        trn_std = np.std(X_trn)

        X_trn = (X_trn - trn_mean) / trn_std

        X_tst = x_test[np.logical_or(y_test == class_a,
                                    y_test == class_b), :]

        X_tst = (X_tst - trn_mean) / trn_std
        return (X_trn, y_trn), (X_tst, y_tst)

# AE related stuff.  Do not touch
def get_relu_class(*_):
    return [ReLU]

def get_linear_class(*_):
    return [Linear]

def get_sigmoid_class(*_):
    return [Sigmoid]

def get_se_loss_class(*_):
    return [SE]


if __name__ == '__main__':
    # set hyperparameters
    learning_rate = 0.3
    batch_size = 5
    N_epochs = 30
    validation_set_fraction = 0.5
    print_each = 30

    # load data - set which two MNIST digits you want to classify
    class_a = 4
    class_b = 9

    (X_trn, y_trn), (X_tst, y_tst) = load_data('full-mnist.npz',
                                               class_a,
                                               class_b)

    N_trn, D = X_trn.shape
    assert D == 28*28
    assert y_trn.shape == (N_trn, 1)

    # Shuffle the dataset
    trn_indices = np.arange(N_trn)
    np.random.shuffle(trn_indices)
    X_trn = X_trn[trn_indices, :]
    y_trn = y_trn[trn_indices, :]

    # Split into train/val
    idx_split = int(np.round(N_trn * validation_set_fraction))

    X_val = X_trn[:idx_split, :]
    X_trn = X_trn[idx_split:, :]
    N_trn = X_trn.shape[0]
    N_val = X_val.shape[0]

    y_val = y_trn[:idx_split, :]
    y_trn = y_trn[idx_split:, :]

    # Now, the magic happens - we build a model...
    model = [Linear(D, 50), ReLU(), Linear(50, 10), ReLU(), Linear(10, 1), Sigmoid()]
    trn_head = SE()

    # And train!
    model_best_params = []
    trn_losses = []
    val_losses = []
    best_val_loss_epoch = 0

    batch_count = int(np.ceil(N_trn / batch_size))
    for epoch in range(N_epochs):
        try:
            cumulative_epoch_trn_loss = 0 # just for reporting progress
            time_start = time.time()
            for batch_i in range(batch_count):
                # load the minibatch:
                batch_idx = range(batch_i * batch_size,
                                  min(N_trn, (batch_i + 1) * batch_size))

                activation = X_trn[batch_idx]

                # forward pass:
                for layer in model:
                    activation = layer.forward(activation)

                loss = trn_head.forward(activation, y_trn[batch_idx])
                trn_loss = loss.mean()

                cumulative_epoch_trn_loss += trn_loss

                # backward pass:
                grad_output = trn_head.backward()
                for layer in reversed(model):
                    grad_output = layer.backward(grad_output)

                # Update the weights with gradient descent
                for layer in model:
                    for param_name, param_value in layer.params().items():
                        param_value -= learning_rate * layer.grads()[param_name].mean(axis=0) # mean across the minibatch

            # validation
            activation = X_val.copy()
            for layer in model:
                activation = layer.forward(activation)
            val_losses.append(trn_head.forward(activation, y_val).mean())

            # remember the best model so far
            if len(val_losses) == 0 or val_losses[-1] < val_losses[best_val_loss_epoch]:
                best_val_loss_epoch = epoch
                model_best_params = [layer.params() for layer in model]

            trn_losses.append(cumulative_epoch_trn_loss / batch_count)
            if epoch % print_each == 0:
                print("[{:04d}][TRN] MSE loss {:2f} ({:.1f}s)".format(epoch, trn_losses[-1], time.time() - time_start))
                print("[{:04d}][VAL] MSE loss {:2f}".format(epoch, val_losses[-1]))
        except KeyboardInterrupt:
            print('Early exit')
            break


    # Plot epochs
    plt.plot(np.arange(len(val_losses)), val_losses)
    plt.plot(np.arange(len(trn_losses)), trn_losses)
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.legend(['validation', 'training'])
    plt.savefig('numpy_nn_training.png')
    plt.show()

    # TST load best model
    print('Best VAL model loss {:.4f} at epoch #{:d}.'.format(val_losses[best_val_loss_epoch], best_val_loss_epoch))
    for layer_id in range(len(model_best_params)):
        for key, value in model_best_params[layer_id].items():
            model[layer_id].params()[key] = value

    # TST forward pass
    activation = X_tst
    for layer in model:
        activation = layer.forward(activation)
    y_hat = (activation > 0.5).astype(np.int)

    loss = trn_head.forward(activation, y_tst)
    print("[TST] MSE loss {:.4f}".format(loss.mean()))

    test_error = np.mean(y_hat != y_tst)
    print("[TST] error {:.4f}".format(test_error))

    plt.figure(figsize=(15, 10))
    plt.title('NN classification: test error {:.4f}'.format(test_error))
    show_classification(X_tst.transpose(1, 0).reshape(28, 28, -1), y_hat.squeeze(), '{}{}'.format(class_a,
                                                                                                  class_b))
    plt.savefig('numpy_nn_classification.png')
    plt.show()
