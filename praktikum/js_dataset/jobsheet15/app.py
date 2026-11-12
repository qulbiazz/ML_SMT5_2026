import pickle
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import tensorflow as tf
from skimage.feature import hog

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "day_night_model.h5"
SCALER_PATH = APP_DIR / "scaler.pkl"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
with SCALER_PATH.open("rb") as scaler_file:
    scaler = pickle.load(scaler_file)


def predict_image(image: np.ndarray) -> dict[str, float]:
    if image is None:
        raise gr.Error("Unggah gambar terlebih dahulu.")

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise gr.Error("Gambar harus memiliki tiga kanal warna (RGB).")
    if np.issubdtype(image.dtype, np.floating) and image.max() <= 1:
        image = image * 255
    image = np.clip(image, 0, 255).astype(np.uint8)

    resized = cv2.resize(image, (256, 256))
    grayscale = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    features = hog(
        grayscale,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True,
    )
    scaled_features = scaler.transform(features.reshape(1, -1))
    day_probability = float(model.predict(scaled_features, verbose=0)[0][0])

    return {
        "Day (Siang)": day_probability,
        "Night (Malam)": 1 - day_probability,
    }


demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy", image_mode="RGB", label="Unggah gambar"),
    outputs=gr.Label(num_top_classes=2, label="Hasil prediksi"),
    title="Day vs Night Classifier",
    description="Unggah gambar untuk mengklasifikasikannya sebagai siang atau malam.",
)

if __name__ == "__main__":
    demo.launch()