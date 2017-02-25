import numpy as np


def sigmoid(x):
    """Calculate sigmoid"""
    return 1 / (1 + np.exp(-x))


x = np.array([0.5, 0.1, -0.2])
target = 0.6
learning_rate = 0.5

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

weights_hidden_output = np.array([0.1, -0.3])

# forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
output = sigmoid(output_layer_in)

# backwards pass
# calculate error
error = target - output

# calculate error gradient for output layer
del_err_output = error * output * (1 - output)
# calculate error gradient for hidden layer
del_err_hidden = np.dot(del_err_output, weights_hidden_output) * hidden_layer_output * (
    1 - hidden_layer_output)
# calculate change in weights for hidden layer to output layer
delta_w_h_o = learning_rate * del_err_output * hidden_layer_output
# calculate change in weights for input layer to hidden layer
delta_w_i_o = learning_rate * del_err_hidden * x[:, None]

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_o)
