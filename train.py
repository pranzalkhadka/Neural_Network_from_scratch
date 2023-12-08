import numpy as np
import pandas as pd

from src.parameter_initialization import ParameterInitialization
from src.forward_propagation import ForwardPropagation
from src.back_propagation import BackPropagation
from src.cost_function import UpdateParameters
from src.evaluation import Evaluation

df = pd.read_csv("data/train.csv")
data = np.array(df)
m,n = data.shape
np.random.shuffle(data)

val_data = data[0 : 2000].T
Y_val = val_data[0]
X_val = val_data[1 : n]
X_val = X_val / 255.

train_data = data[2000 : m].T
Y_train = train_data[0]
X_train = train_data[1 : n]
X_train = X_train / 255.

epochs = 500
learning_rate = 0.10


def run(X, Y, alpha, epochs):
    parameter_init = ParameterInitialization()
    forward_propagation = ForwardPropagation()
    back_propagation = BackPropagation()
    update_parameters = UpdateParameters()
    evaluation = Evaluation()

    W1, b1, W2, b2, W3, b3 = parameter_init.initialize_parameters()
    for i in range(epochs):
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation.forward_propagation(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = back_propagation.back_propagation(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_parameters.update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        #Print the accuracy at every 10th epoch
        if i % 10 == 0:
            print("Epoch:", i)
            prediction = evaluation.predictions(A3)
            print(f"Accuracy: {evaluation.accuracy(prediction, Y)}")
            print("~~~~~~~~~~~~~~~~~")

    np.savez("trained_parameters.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
    
    return W1, b1, W2, b2, W3, b3

W1, b1, W2, b2, W3, b3 = run(X_train, Y_train, learning_rate, epochs)
