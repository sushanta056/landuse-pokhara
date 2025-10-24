import os
import joblib
import numpy as np
import rasterio
from rasterio.windows import Window
from flask import Flask, request, render_template, send_file

app = Flask(__name__)

# Limit upload size to 200 MB
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

# Path to model
MODEL_PATH = "modelxgb.joblib"

# Load model
model = joblib.load(MODEL_PATH)
print(f"Loaded model: {MODEL_PATH}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    input_path = "input.tif"
    output_path = "predicted_landuse.tif"

    file.save(input_path)

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        profile.update(dtype=rasterio.uint8, count=1, compress="lzw")

        # Create output file
        with rasterio.open(output_path, "w", **profile) as dst:
            tile_size = 512  # process 512x512 chunks
            for y in range(0, src.height, tile_size):
                for x in range(0, src.width, tile_size):
                    window = Window(x, y,
                                    min(tile_size, src.width - x),
                                    min(tile_size, src.height - y))

                    data = src.read(window=window)
                    feature_stack = np.moveaxis(data, 0, -1)
                    flat_features = feature_stack.reshape(-1, feature_stack.shape[-1])

                    # Predict only on valid pixels
                    valid_mask = np.all(np.isfinite(flat_features), axis=1)
                    preds = np.zeros(flat_features.shape[0], dtype=np.uint8)

                    if np.any(valid_mask):
                        preds[valid_mask] = model.predict(flat_features[valid_mask])

                    pred_2d = preds.reshape(feature_stack.shape[:2]).astype(np.uint8)
                    dst.write(pred_2d, 1, window=window)

    return send_file(output_path, as_attachment=True)

if __name__ == "__main__":
    # Debug for local use only
    app.run(host="0.0.0.0", port=5000, debug=True)

