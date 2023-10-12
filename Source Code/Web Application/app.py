from flask import Flask, render_template, url_for, request, redirect, make_response
import requests
import random
import os
import os
import sys
import subprocess

app = Flask(__name__)
app.secret_key = "secret key"
UPLOAD_FOLDER = "static/images/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
@app.route("/home")
def home():
    for file in os.listdir("static/images/"):
        tmp = os.path.join("static/images/", file)
        os.remove(tmp)
    return render_template("home.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["photo"]
    tmp = file.read()
    files = {"file": tmp}
    url = "http://192.168.1.9:8080/predict"
    response = requests.post(url, files=files)
    data = response.json()

    tmpFileName = str(random.randint(1, 100000000)) + ".jpg"

    with open(f"static/images/{tmpFileName}", "wb") as f:
        f.write(tmp)

    return render_template("success.html", data=data, filename=tmpFileName)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
