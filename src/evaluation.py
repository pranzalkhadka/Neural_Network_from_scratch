import numpy as np
from src.forward_propagation import ForwardPropagation

forward_propagation = ForwardPropagation()

class Evaluation:
    """
    This class defines some functions to evaluate the trained model
    """

    def predictions(self, A3):
        return np.argmax(A3, 0)
    

    def accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size
    

    def make_predictions(self, X, W1, b1, W2, b2, W3, b3):
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation.forward_propagation(W1, b1, W2, b2, W3, b3, X)
        prediction = self.predictions(A3)
        return prediction
    