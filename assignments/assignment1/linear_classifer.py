import numpy as np


def _softmax(z):
    # z -= np.max(z)
    return np.exp(z - np.max(z)) / (np.sum(np.exp(z - np.max(z))))

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    if len(predictions.shape) == 1:
        return _softmax(predictions)
    else:
        return np.apply_along_axis(_softmax, 1, predictions)


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    y_true = np.zeros(probs.shape)
    if isinstance(target_index, int):
        y_true[target_index] = 1
    else:
        count = 0
        for sample_y_true in target_index:
            y_true[count][sample_y_true] = 1
            count += 1

    return - np.sum(y_true * np.log(probs))


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    y_true = np.zeros(predictions.shape)
    if isinstance(target_index, int):
        y_true[target_index] = 1
    else:
        count = 0
        for sample_y_true in target_index:
            y_true[count][sample_y_true] = 1
            count += 1

    softmax_prob = softmax(predictions)
    loss = cross_entropy_loss(softmax_prob, target_index)

    dprediction = softmax_prob - y_true

    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    f = lambda x: reg_strength * np.sum(np.square(x))

    loss = f(W)

    it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])

    grad = W.copy()
    while not it.finished:
        ix = it.multi_index

        bigger_x = W.copy()
        bigger_x[ix] += reg_strength
        lesser_x = W.copy()
        lesser_x[ix] -= reg_strength

        numeric_grad_at_ix = (f(bigger_x) - f(lesser_x)) / (2 * reg_strength)
        grad[ix] = numeric_grad_at_ix
        it.iternext()

    return loss, grad
    


def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W

    loss, dW = softmax_with_cross_entropy(predictions, target_index)

    dW = X.transpose().dot(dW)

    return loss, dW

class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            loss_epoch = np.array([])
            for batch in batches_indices:
                loss, dW = linear_softmax(X[batch], self.W, y[batch])
                loss_reg, dW_reg = l2_regularization(self.W, reg)
                self.W += - learning_rate * (dW + dW_reg)
                loss_epoch = np.append(loss_epoch, loss)
            loss_history.append(loss_epoch.mean())
            # end
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)
        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        predictions = np.dot(X, self.W)
        probs = softmax(predictions)
        y_pred = probs.argmax(axis=1)

        return y_pred
