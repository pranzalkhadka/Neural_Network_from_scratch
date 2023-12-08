import numpy as np
import pandas as pd
from src.activation_function import ActivationFunction
from src.one_hot_encoding import OneHotEncoding

activation_function = ActivationFunction()
one_hot_encoding = OneHotEncoding()

df = pd.read_csv("data/train.csv")
data = np.array(df)
m,n = data.shape

class BackPropagation:
    
    """
    This class uses Derivative chain rule to perform back propagation
    """

    def back_propagation(self, Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
        one_hot_Y = one_hot_encoding.one_hot_encoding(Y)
    
        dZ3 = A3 - one_hot_Y
        dW3 = 1 / m * dZ3.dot(A2.T)
        db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True) 
    
        dZ2 = W3.T.dot(dZ3) * activation_function.derivative_ReLU(Z2)
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    
        dZ1 = W2.T.dot(dZ2) * activation_function.derivative_ReLU(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

        return dW1, db1, dW2, db2, dW3, db3