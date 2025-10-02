import os
import numpy as np
import torch
import tifffile
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image

import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt


UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "static/results"
ALLOWED_EXTENSIONS = {"tif", "tiff"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=12,
    classes=1,
).to(device)

model.load_state_dict(torch.load("water_segmentation_unet_resnet34.pth", map_location=device))
model.eval() # evaluation mode (disables dropout, BN updates).

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_mask(tif_path, result_path):
    img = tifffile.imread(tif_path).astype(np.float32)
    img /= img.max()

    X_input = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).to(device).float()

    with torch.no_grad():
        y_pred = model(X_input)
        y_pred = torch.sigmoid(y_pred)
        y_pred = (y_pred > 0.5).float().cpu().numpy().squeeze()

    # Save visualization
    plt.figure(figsize=(12, 4))

    # Input (RGB approx)
    rgb_image = img[:, :, :3]
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-6)
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_image)
    plt.title("Input (RGB approx)")
    plt.axis("off")

    # Predicted mask
    plt.subplot(1, 3, 2)
    plt.imshow(y_pred, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(rgb_image)
    plt.imshow(y_pred, cmap="jet", alpha=0.4)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(result_path)
    plt.close()



@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            result_path = os.path.join(app.config["RESULT_FOLDER"], f"result_{filename}.png")
            predict_mask(filepath, result_path)

            return render_template("index.html", result_image=url_for("static", filename=f"results/result_{filename}.png"))

    return render_template("index.html", result_image=None)


if __name__ == "__main__":
    app.run(port=3000, debug=True)
