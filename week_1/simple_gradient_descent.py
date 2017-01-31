import numpy as np


def sigmoid(x):
    """Calculate sigmoid"""
    return 1 / (1 + np.exp(-x))


learning_rate = 0.5
x = np.array([1, 2])
y = np.array(0.5)

# initial weights
w = np.array([0.5, -0.5])

# calculate one gradient descent step for each weight
# calculate output of neural network
nn_output = sigmoid(np.dot(w, x))

# calculate error of neural network
error = y - nn_output

# calculate change in weights
del_w = [learning_rate * error * nn_output * (1 - nn_output) * x[0],
         learning_rate * error * nn_output * (1 - nn_output) * x[1]]

print('Neural Network output:')
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)
