from flask import Flask, request, render_template, redirect
from PIL import Image
import torch as t
from utils import predict
from torchvision import models
import os

app = Flask(__name__)
app.secret_key = "ABCDE"
UPLOAD_FOLDER = "static/uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

model = models.resnet50()
model.fc = t.nn.Linear(model.fc.in_features, 100)
model.load_state_dict(t.load("./model_25092024-094302.pth"))
model.eval()

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)

        uploaded_image = request.files["image"]

        if uploaded_image.filename == "":
            return redirect(request.url)
        
        if uploaded_image:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_image.filename)
            uploaded_image.save(filepath)
            image = Image.open(uploaded_image)
            label = predict(
                model=model,
                image=image,
            )
            return render_template("index.html", image=uploaded_image.filename, label=label)

    clear_images(app.config["UPLOAD_FOLDER"])

    return render_template("index.html", image=None)

def clear_images(upload_folder):
    for file in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    app.run(debug=True)