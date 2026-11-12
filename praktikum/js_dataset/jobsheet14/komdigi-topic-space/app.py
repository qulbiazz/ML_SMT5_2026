import pickle
from pathlib import Path

import gradio as gr
import numpy as np
import tensorflow as tf

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "komdigi_topic_model.h5"
PREPROCESSOR_PATH = APP_DIR / "komdigi_preprocessor.pkl"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
with PREPROCESSOR_PATH.open("rb") as file:
    preprocessor = pickle.load(file)

vectorizer = preprocessor["vectorizer"]
label_encoder = preprocessor["label_encoder"]


def predict_topic(title: str, article_text: str) -> dict[str, float]:
    title = title or ""
    article_text = article_text or ""
    text = f"{title.strip()} {article_text.strip()}".strip()
    if not text:
        raise gr.Error("Masukkan judul atau isi artikel terlebih dahulu.")

    features = vectorizer.transform([text]).toarray().astype(np.float32)
    probabilities = model.predict(features, verbose=0)[0]
    return {
        str(label): float(probability)
        for label, probability in zip(label_encoder.classes_, probabilities)
    }


demo = gr.Interface(
    fn=predict_topic,
    inputs=[
        gr.Textbox(label="Judul artikel", placeholder="Masukkan judul artikel"),
        gr.Textbox(
            label="Isi artikel",
            lines=8,
            placeholder="Tempelkan isi artikel atau ringkasannya",
        ),
    ],
    outputs=gr.Label(num_top_classes=3, label="Prediksi topik"),
    title="Klasifikasi Topik Artikel Komdigi",
    description=(
        "Model memprediksi topik artikel berdasarkan data klarifikasi Komdigi. "
        "Hasilnya adalah kategori topik, bukan verifikasi apakah artikel itu hoaks."
    ),
)

if __name__ == "__main__":
    demo.launch()
