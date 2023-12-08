from src.activation_function import ActivationFunction

activation_function = ActivationFunction()

class ForwardPropagation:
    
    """
    This class is responsible for passing our data points through the neural network
    It uses ReLU for hidden layers and softmax for final output layer as activation function
    """

    def forward_propagation(self, W1, b1, W2, b2, W3, b3, X):
        Z1 = W1.dot(X) + b1
        A1 = activation_function.ReLU(Z1)

        Z2 = W2.dot(A1) + b2
        A2 = activation_function.ReLU(Z2)

        Z3 = W3.dot(A2) + b3
        A3 = activation_function.Softmax(Z3)

        return Z1, A1, Z2, A2, Z3, A3