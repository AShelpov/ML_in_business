import os
import numpy as np
from keras import models
from keras.preprocessing.image import img_to_array
from PIL import Image

import flask
import io


app = flask.Flask(__name__)
model = None


def load_model():
    global model
    model = models.load_model(os.path.join(os.getcwd(), "model"))


def prepare_image(image):
    image = image.resize((48,48))
    image = image.convert(mode="L")
    image = img_to_array(image)
    image = image.astype('float32')
    image /= 255
    image = np.expand_dims(image, axis=0)
    image = image.reshape(1, 48, 48, 1)
    return image


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image)
            print("Image have benn successfully prepared")
            # predict label
            result = model.predict(image)

            label_dict = {0: 'angry', 1: 'disgusted', 2: 'fearful', 3: 'happy',
                          4: 'neutral', 5: 'sad', 6: 'surprised'}
            predict_label = label_dict[np.argmax(result[0])]
            print("Output have been successfully predicted")
            # indicate that the request was a success
            data["success"] = True
            data["predicted_label"] = predict_label

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run()

