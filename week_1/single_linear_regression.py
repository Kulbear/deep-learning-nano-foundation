from numpy import *


def compute_error(b, m, points):
    """Compute MSE by given model

    :param b: scalar
    :param m: intercept
    :param points: dataset list
    :return: mean squared error
    """
    sum_error = 0
    N = len(points)

    # compute the sum of squared residuals
    for i in range(N):
        x = points[i, 0]
        y = points[i, 1]
        sum_error += (y - (m * x + b)) ** 2

    # mean squared error
    return sum_error / N


def step_gradient(current_b, current_m, points, learning_rate=0.0001):
    """The gradient step.

    This function update intercept and scalar by using the partial
    derivative of the cost function wrt to b and m.

    :param current_b: current scalar error
    :param current_m: current intercept
    :param points: dataset list
    :param learning_rate: convergence speed
    :return: updated intercept and updated scalar
    """
    gradient_b = 0
    gradient_m = 0
    N = len(points)

    for i in range(N):
        x = points[i, 0]
        y = points[i, 1]

        # partial derivative of the cost function wrt b and m
        gradient_b += -(2 / N) * (y - (current_m * x + current_b))
        gradient_m += -(2 / N) * x * (y - (current_m * x + current_b))

        new_b = current_b - (learning_rate * gradient_b)
        new_m = current_m - (learning_rate * gradient_m)

    return new_b, new_m


def gradient_decent_runner(points, starting_b=0, starting_m=0,
                           learning_rate=0.0001,
                           num_iteration=10000):
    """The gradient descent calculation process

    Results are calculated by gradient steps
    and should converge to optimal (according to number of iteration).

    :param points: dataset list
    :param starting_b: starting scalar
    :param starting_m: starting intercept
    :param learning_rate: convergence speed
    :param num_iteration: number of gradient descent steps
    :return: result intercept and scalar
    """
    # initialization
    b = starting_b
    m = starting_m

    # gradient descent
    for i in range(num_iteration + 1):
        # update b and m
        if i % 100 == 0:
            print('After {} iteration, b = {}, m = {}'.format(i, b, m))
        b, m = step_gradient(b, m, array(points), learning_rate)

    return [b, m]


def run(path, learning_rate=0.0001, initial_b=0, initial_m=0,
        num_iteration=10000):
    """Run a single linear regression process

    :param path: csv file path
    :param learning_rate: convergence speed
    :param initial_b: initial value of scalar
    :param initial_m: initial value of intercept
    :param num_iteration: number of gradient descent steps
    :return: None
    """
    points = genfromtxt(path, delimiter=',')

    print('Starting gradient descent at b = {}, m = {}, error = {}'.format(
        initial_b, initial_m, compute_error(initial_b, initial_m, points)))
    [b, m] = gradient_decent_runner(points, initial_b, initial_m, learning_rate, num_iteration)

    print('Ending point at b = {}, m = {}, error = {}'.format(b, m, compute_error(b, m, points)))


if __name__ == '__main__':
    data_path = 'data.csv'
    learning_rate = 0.0001
    initial_b = 0
    initial_m = 0
    num_iteration = 10000
    run(data_path, learning_rate, initial_b, initial_m, num_iteration)
