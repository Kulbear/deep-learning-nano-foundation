import numpy as np
from data_prep import features, targets, features_test, targets_test


def sigmoid(x):
    """Calculate sigmoid"""
    return 1 / (1 + np.exp(-x))


# use to same seed to make debugging easier
np.random.seed(42)

n_records, n_features = features.shape
last_loss = None

# initialize weights
weights = np.random.normal(scale=1 / n_features ** .5, size=n_features)

# neural network hyperparameters
epochs = 1000
learnrate = 0.5

for e in range(epochs):
    del_w = np.zeros(weights.shape)

    # Loop through all records, x is the input, y is the target
    for x, y in zip(features.values, targets):
        # forward pass
        output = sigmoid(np.dot(weights, x))
        # calculate error
        error = (y - output) * output * (1 - output)
        # update delta_w
        del_w += error * x

    weights += learnrate * del_w / n_records

    # printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        out = sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# calculate accuracy on test data
tes_out = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
