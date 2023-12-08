import numpy as np

class ParameterInitialization:

    """
    This function initializes random values for initial weights and biases of our neural network
    There will be a input layer, two hidden layer with 300 and 100 neurons and output layer with 10 neurons representing mnist digits
    """

    def initialize_parameters(self):
        W1 = np.random.rand(300, 784) - 0.5
        #returns an array of random values between (0, 1) with the shape (300, 784)
        b1 = np.random.rand(300, 1) - 0.5
        #returns an array of random values between (0, 1) with the shape (300, 1)

        W2 = np.random.rand(100, 300) - 0.5
        #returns an array of random values between (0, 1) with the shape (100, 300)
        b2 = np.random.rand(100, 1) - 0.5
        #returns an array of random values between (0, 1) with the shape (100, 1)
        
        W3 = np.random.rand(10, 100) - 0.5
        #returns an array of random values between (0, 1) with the shape (10, 100)
        b3 = np.random.rand(10, 1) - 0.5
        #returns an array of random values between (0, 1) with the shape (10, 1)

        return W1, b1, W2, b2, W3, b3