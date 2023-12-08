import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

df = pd.read_csv("data/train.csv")
data = np.array(df)
m, n = data.shape

np.random.shuffle(data)
val_data = data[0:2000].T
Y_val = val_data[0]
X_val = val_data[1:n]
X_val = X_val / 255.

train_data = data[2000:m].T
Y_train = train_data[0]
X_train = train_data[1:n]
X_train = X_train / 255.

from src.evaluation import Evaluation
evaluation = Evaluation()

from fastapi import FastAPI
import uvicorn

app = FastAPI()

weights = np.load("trained_parameters.npz")
W1, b1, W2, b2, W3, b3 = weights['W1'], weights['b1'], weights['W2'], weights['b2'], weights['W3'], weights['b3']

def plot_to_base64(index, current_image, prediction, label):
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation="nearest")
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

@app.get('/')
def message():
    return {'FastAPI application for Inference'}

@app.post("/predict/{index}")
def predict_mnist(index: int):
    current_image = X_train[:, index, None]
    prediction = evaluation.make_predictions(
        X_train[:, index, None], W1, b1, W2, b2, W3, b3
    )
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    pil_image = plot_to_base64(index, current_image, prediction, label)
    img_byte_array = io.BytesIO()
    pil_image.save(img_byte_array, format="PNG")
    img_byte_array = img_byte_array.getvalue()

    return StreamingResponse(io.BytesIO(img_byte_array), media_type="image/png")


if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

