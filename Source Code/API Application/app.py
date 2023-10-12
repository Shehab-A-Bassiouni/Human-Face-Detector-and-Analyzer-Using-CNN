from flask import Flask, request, jsonify
import os
from model import Model
import random

app = Flask(__name__)
app.secret_key = "secret key"
UPLOAD_FOLDER = "static/images/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

reply = {
    "Gender": "NULL",
    "Age": "NULL",
    "Ethnicity": "NULL",
    "Emotions": "NULL",
}


@app.route("/predict", methods=["GET", "POST"])
def getPredict():
    file = request.files["file"]

    filename = str(random.randint(1, 100000000)) + file.filename
    full_filename = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(full_filename)
    obj = Model(full_filename)
    gender = obj.Gender_Predict()
    age = obj.Age_Predict()
    ethnicity = obj.Ethnicity_Predict()
    emotion = obj.Emotions_Predict()
    os.remove(full_filename)
    reply["Gender"] = gender
    reply["Age"] = age
    reply["Ethnicity"] = ethnicity
    reply["Emotions"] = emotion
    return jsonify(reply)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
