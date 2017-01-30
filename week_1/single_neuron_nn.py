from numpy import *


class NeuralNetwork:
    def __init__(self):
        # set a fixed random seed to produce same result for each run
        random.seed(1)

        # model a single neuron, with 4 input connections and 1 output connection
        # random weight 3 x 1 matrix, with values in range -1 to 1, mean 0
        self.weights = 2 * random.random((4, 1)) - 1

    def __sigmoid(self, x):
        """The sigmoid curve function."""
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        """The gradient of the sigmoid curve."""
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, num_iteration):
        """Train the neural network.

        Use a simple backpropagation algorithm to update the weights.
        """
        for iteration in range(num_iteration):
            output = self.predict(training_set_inputs)

            # calculate the error
            error = training_set_outputs - output

            # multiply the error by the input ad again by the gradient of the sigmoid curve
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # adjust the weights
            self.weights += adjustment

    def predict(self, inputs):
        """Pass out inputs to the neural network."""
        return self.__sigmoid(dot(inputs, self.weights))


if __name__ == "__main__":
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights:\n{}".format(neural_network.weights))

    # The training set. We have 4 examples, each consisting of 4 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1, 1], [1, 1, 1, 1], [1, 0, 1, 0], [0, 1, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("New synaptic weights after training:\n{}".format(neural_network.weights))

    # Test the neural network with new situations.
    print("Considering a new situation [1, 1, 0, 1] -> ?: ")
    print(neural_network.predict(array([1, 1, 0, 1])))
    print("Considering a new situation [0, 0, 0, 1] -> ?: ")
    print(neural_network.predict(array([0, 0, 0, 1])))
    print("Considering a situation included in our training set [0, 0, 1, 1] -> ?: ")
    print(neural_network.predict(array([0, 0, 1, 1])))
