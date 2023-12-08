import numpy as np
import pandas as pd

from src.evaluation import Evaluation
evaluation = Evaluation()

df = pd.read_csv("data/train.csv")
data = np.array(df)
m,n = data.shape
np.random.shuffle(data)

val_data = data[0 : 2000].T
Y_val = val_data[0]
X_val = val_data[1 : n]
X_val = X_val / 255.

weights = np.load("trained_parameters.npz")
W1, b1, W2, b2, W3, b3 = weights['W1'], weights['b1'], weights['W2'], weights['b2'], weights['W3'], weights['b3']

val_prediction = evaluation.make_predictions(X_val, W1, b1, W2, b2, W3, b3)
print(f"The validation accuracy is {evaluation.accuracy(val_prediction, Y_val) * 100}")