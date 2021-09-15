from flask import render_template, redirect
import argparse
import io
from PIL import Image
import torch
from flask import Flask, request
import os


application = Flask(__name__)

model = torch.hub.load(".", "custom", path='./yolov5s.pt', source='local').autoshape()
model.eval()


@application.route("/", methods=["GET", "POST"])
def predict():
    if len(request.files['file'].filename) != 0:
        if request.method == "POST":
            file = request.files["file"]
            image_bytes = file.read()
            image_name = file.filename

            filename, file_extension = os.path.splitext(image_name)

            if file_extension in ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng', '.webp', '.mpo']:
                img = Image.open(io.BytesIO(image_bytes))
                results = model(img, size=640)
                results.render()
                results.save('./static/')

            return render_template("return.html")
        return render_template("index.html")
    else:
        return render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask web_page for object detection")
    parser.add_argument("--port", default=80, type=int, help="port number")

    args = parser.parse_args()

    application.run(host="0.0.0.0", port=args.port, debug=False)  # debug=True causes Restarting with stat
