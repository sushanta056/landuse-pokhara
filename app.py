# app.py
import os
import tempfile
import numpy as np
import joblib
import rasterio
from flask import Flask, request, send_file, jsonify, render_template

MODEL_PATH = "modelxgb.joblib"  # update: no "models/" folder
OUT_FILENAME = "predicted_landuse.tif"

app = Flask(__name__)

# Load the model once at startup
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print("Loaded model:", MODEL_PATH)


def predict_geotiff(in_path, out_path):
    # Read and predict GeoTIFF in small tiles
    with rasterio.open(in_path) as src:
        meta = src.meta.copy()
        height, width = src.height, src.width

        meta.update({
            "count": 1,
            "dtype": rasterio.uint8,
            "compress": "lzw"
        })

        # Create output file
        with rasterio.open(out_path, "w", **meta) as dst:
            tile_size = 512  # process 512x512 chunks
            for y in range(0, height, tile_size):
                for x in range(0, width, tile_size):
                    window = rasterio.windows.Window(
                        x, y,
                        min(tile_size, width - x),
                        min(tile_size, height - y)
                    )

                    # Read required 9 bands in order
                    b_B11 = src.read(5, window=window).astype("float32")
                    b_B12 = src.read(6, window=window).astype("float32")
                    b_B2  = src.read(1, window=window).astype("float32")
                    b_B3  = src.read(2, window=window).astype("float32")
                    b_B4  = src.read(3, window=window).astype("float32")
                    b_B8  = src.read(4, window=window).astype("float32")
                    b_NDBI= src.read(9, window=window).astype("float32")
                    b_NDVI= src.read(7, window=window).astype("float32")
                    b_NDWI= src.read(8, window=window).astype("float32")

                    # Stack and reshape for prediction
                    feature_stack = np.stack([
                        b_B11, b_B12, b_B2, b_B3, b_B4,
                        b_B8, b_NDBI, b_NDVI, b_NDWI
                    ], axis=0)

                    n_features = feature_stack.shape[0]
                    flat = feature_stack.reshape(n_features, -1).T
                    valid_mask = np.any(np.isfinite(feature_stack) & (feature_stack != 0), axis=0)
                    valid_flat = valid_mask.reshape(-1)
                    flat_preds = np.full((flat.shape[0],), fill_value=4, dtype=np.uint8)

                    if np.any(valid_flat):
                        preds = model.predict(flat[valid_flat])
                        flat_preds[valid_flat] = preds.astype(np.uint8)

                    pred_map = flat_preds.reshape(feature_stack.shape[1:])
                    dst.write(pred_map, 1, window=window)


@app.route("/")
def index():
    # Serve the HTML upload form
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "tif" not in request.files:
        return jsonify({"error": "No 'tif' file sent"}), 400

    tif_file = request.files["tif"]
    if tif_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded file to temp file
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_in:
        tmp_in_name = tmp_in.name
        tif_file.save(tmp_in_name)

    tmp_out = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    tmp_out_name = tmp_out.name
    tmp_out.close()

    try:
        predict_geotiff(tmp_in_name, tmp_out_name)
        return send_file(tmp_out_name,
                         as_attachment=True,
                         download_name=OUT_FILENAME,
                         mimetype="image/tiff")
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(tmp_in_name)
        except Exception:
            pass
        # keep tmp_out to let send_file complete


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
