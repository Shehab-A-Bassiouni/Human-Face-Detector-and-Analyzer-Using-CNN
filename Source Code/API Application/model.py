import tensorflow as tf
import cv2
import numpy as np
import keras
import keras_vggface


class Model:
    def __init__(self, image_path) -> None:
        self.Gender_Model = tf.keras.models.load_model("Gender.h5")
        self.Age_Model = tf.keras.models.load_model("Age.h5")
        self.Ethnicity_Model = keras.models.load_model("Ethnicity.h5")
        self.Emotions_Model = tf.keras.models.load_model("Emotions.h5", compile=False)
        self.Emotions_Model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer="adam",
            metrics=["accuracy"],
        )

        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.img = cv2.imread(image_path)
        self.crop()

    def crop(self):
        faces = self.face_cascade.detectMultiScale(
            self.img, scaleFactor=1.2, minNeighbors=6, minSize=(100, 100)
        )
        for i, (x, y, w, h) in enumerate(faces):
            x2, y2, w2, h2 = self.shrink(x, y, w, h)
            self.img = self.img[y2 : y2 + h2, x2 : x2 + w2]
            break

    def shrink(self, x, y, w, h, scale=0.9):
        wh_multiplier = (1 - scale) / 2
        x_new = int(x + (w * wh_multiplier))
        y_new = int(y + (h * wh_multiplier))
        w_new = int(w * scale)
        h_new = int(h * scale)
        return (x_new, y_new, w_new, h_new)

    def Emotions_Predict(self):
        Emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        Emo = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        Emo = cv2.resize(Emo, (48, 48))
        Emo = Emo.reshape(-1, 48, 48, 1)
        Emo = Emo / 255.0
        pred = self.Emotions_Model.predict(Emo)
        Pred_Emo = Emotions[np.argmax(pred)]
        return Pred_Emo

    def Ethnicity_Predict(self):
        ethnicity_ranges = ["caucassian", "African", "Indian", "Asian", "middle east"]
        E_img = self.img.astype("float64")
        E_img = keras_vggface.utils.preprocess_input(E_img)
        E_img = cv2.resize(E_img, (224, 224))
        E_img = E_img.reshape(-1, 224, 224, 3)
        E_img = E_img / 255.0
        pred = self.Ethnicity_Model.predict(E_img)
        Ethnicity = ethnicity_ranges[np.argmax(pred)]
        return Ethnicity

    def Age_Predict(self):
        age_ranges = ["1-2", "3-9", "10-20", "21-27", "28-45", "46-65", "66-116"]
        A_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        A_img = cv2.resize(A_img, (200, 200))
        A_img = A_img.reshape(-1, 200, 200, 1)
        pred = self.Age_Model.predict(A_img)
        age = age_ranges[np.argmax(pred)]
        return age

    def Gender_Predict(self):
        G_img = cv2.resize(self.img, (128, 128))
        G_img = G_img.reshape(-1, 128, 128, 3)
        G_img = G_img / 255.0
        Gender_prediction = self.Gender_Model.predict(G_img)
        print(Gender_prediction[0][0])
        prediction = "Female"
        if Gender_prediction >= 0.5:
            prediction = "Male"
        return prediction
