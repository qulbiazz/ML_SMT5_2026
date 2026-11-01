import base64
import pickle
import numpy as np
import cv2
import tensorflow as tf

from pathlib import Path
from flask import Flask, request, render_template_string
from skimage.feature import hog


# ==========================================
# 1. INISIALISASI FLASK
# ==========================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "day_night_model.h5"
SCALER_PATH = BASE_DIR / "scaler.pkl"


# ==========================================
# 2. MEMUAT MODEL DAN SCALER
# ==========================================

try:
    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    print("\n" + "=" * 45)
    print("SYSTEM LOADED SUCCESSFULLY")
    print(f"Model       : {MODEL_PATH}")
    print(f"Scaler      : {SCALER_PATH}")
    print(f"Model input : {model.input_shape}")
    print(f"Model output: {model.output_shape}")
    print(f"Scaler fitur: {scaler.n_features_in_}")
    print("=" * 45 + "\n")

except Exception as e:
    print(f"Gagal memuat model atau scaler: {e}")
    raise


# ==========================================
# 3. TEMPLATE HTML
# ==========================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Day & Night Detection</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            font-family: Arial, sans-serif;
            background: #f1f5f9;
            margin: 0;
            padding: 40px 20px;
            text-align: center;
            color: #1e293b;
        }

        .container {
            background: white;
            max-width: 500px;
            margin: auto;
            padding: 30px;
            border-radius: 16px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }

        h1 {
            color: #1e293b;
        }

        p {
            line-height: 1.6;
        }

        input[type="file"] {
            display: block;
            width: 100%;
            margin: 20px 0;
            padding: 10px;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
        }

        button {
            background: #2563eb;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
        }

        button:hover {
            background: #1d4ed8;
        }

        .result {
            margin-top: 24px;
            padding: 16px;
            background: #eff6ff;
            border-radius: 10px;
            overflow-wrap: anywhere;
        }

        .result img {
            max-width: 100%;
            max-height: 300px;
            margin-top: 16px;
            border-radius: 8px;
        }

        .error {
            margin-top: 20px;
            padding: 12px;
            background: #fef2f2;
            color: #dc2626;
            border-radius: 8px;
            overflow-wrap: anywhere;
        }
    </style>
</head>

<body>
    <div class="container">
        <h1>Day & Night Detection</h1>

        <p>
            Unggah gambar untuk mendeteksi apakah gambar
            menunjukkan suasana siang atau malam.
        </p>

        <form method="POST" enctype="multipart/form-data">
            <input
                type="file"
                name="image"
                accept="image/*"
                required
            >

            <button type="submit">Deteksi Gambar</button>
        </form>

        {% if result %}
        <div class="result">
            <h2>Hasil: {{ result }}</h2>

            <p>
                Skor prediksi model:
                {{ "%.6f"|format(score) }}
            </p>

            <img
                src="data:image/jpeg;base64,{{ image_data }}"
                alt="Gambar yang diunggah"
            >
        </div>
        {% endif %}

        {% if error %}
        <div class="error">
            {{ error }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


# ==========================================
# 4. ROUTE DAN PROSES PREDIKSI
# ==========================================

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    score = None
    image_data = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if uploaded_file is None or uploaded_file.filename == "":
            error = "Silakan pilih gambar terlebih dahulu."

        else:
            try:
                # ----------------------------------
                # A. Membaca gambar
                # ----------------------------------

                image_bytes = uploaded_file.read()

                if not image_bytes:
                    raise ValueError("File gambar kosong.")

                image_array = np.frombuffer(
                    image_bytes,
                    dtype=np.uint8
                )

                image = cv2.imdecode(
                    image_array,
                    cv2.IMREAD_COLOR
                )

                if image is None:
                    raise ValueError(
                        "File tidak dapat dibaca sebagai gambar."
                    )

                # ----------------------------------
                # B. Preprocessing
                # ----------------------------------

                # Samakan preprocessing dengan notebook training
                resized = cv2.resize(image, (256, 256))

                # BGR ke RGB, sesuai notebook
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

                # RGB ke grayscale
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

                # ----------------------------------
                # C. Ekstraksi fitur HOG
                # ----------------------------------

                features = hog(
                    gray,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    block_norm="L2-Hys",
                    visualize=False,
                    feature_vector=True
                )

                features = features.reshape(1, -1)

                # Pastikan jumlah fitur sesuai scaler
                if features.shape[1] != scaler.n_features_in_:
                    raise ValueError(
                        "Jumlah fitur HOG tidak sesuai dengan scaler. "
                        f"Fitur gambar: {features.shape[1]}, "
                        f"fitur scaler: {scaler.n_features_in_}. "
                        "Periksa preprocessing saat training."
                    )

                # ----------------------------------
                # D. Normalisasi fitur
                # ----------------------------------

                features_scaled = scaler.transform(features)

                # Pastikan bentuk fitur sesuai input model
                expected_features = model.input_shape[-1]

                if features_scaled.shape[1] != expected_features:
                    raise ValueError(
                        "Jumlah fitur tidak sesuai dengan input model. "
                        f"Fitur: {features_scaled.shape[1]}, "
                        f"input model: {expected_features}."
                    )

                # ----------------------------------
                # E. Prediksi
                # ----------------------------------

                prediction = model.predict(
                    features_scaled,
                    verbose=0
                )

                prediction_array = np.asarray(prediction)

                if prediction_array.size != 1:
                    raise ValueError(
                        "Output model bukan satu nilai. "
                        f"Bentuk output: {prediction_array.shape}. "
                        "Periksa arsitektur dan jenis model."
                    )

                score = float(prediction_array.reshape(-1)[0])

                if not np.isfinite(score):
                    raise ValueError(
                        "Model menghasilkan skor yang tidak valid."
                    )

                # ----------------------------------
                # F. DEBUG
                # ----------------------------------

                print("\n========== DEBUG PREDIKSI ==========")
                print("Nama file       :", uploaded_file.filename)
                print("Ukuran gambar   :", image.shape)
                print("Bentuk fitur HOG:", features.shape)
                print("Bentuk fitur    :", features_scaled.shape)
                print("Input model     :", model.input_shape)
                print("Output model    :", model.output_shape)
                print("Prediksi mentah :", prediction_array)
                print("Skor prediksi   :", score)
                print("====================================\n")

                # ----------------------------------
                # G. Menentukan kelas
                # ----------------------------------
                # Asumsi awal:
                # output > 0.5 = Day
                # output <= 0.5 = Night
                #
                # Pemetaan ini HARUS cocok dengan label
                # yang digunakan saat training.

                result = (
                    "Day (Siang)"
                    if score > 0.5
                    else "Night (Malam)"
                )

                # ----------------------------------
                # H. Menyiapkan gambar untuk HTML
                # ----------------------------------

                success, encoded_image = cv2.imencode(
                    ".jpg",
                    image
                )

                if not success:
                    raise ValueError(
                        "Gagal memproses gambar untuk ditampilkan."
                    )

                image_data = base64.b64encode(
                    encoded_image.tobytes()
                ).decode("utf-8")

            except Exception as e:
                app.logger.exception(
                    "Terjadi kesalahan saat memproses gambar."
                )

                error = f"Gagal memproses gambar: {e}"

    return render_template_string(
        HTML_TEMPLATE,
        result=result,
        score=score,
        image_data=image_data,
        error=error
    )


# ==========================================
# 5. MENJALANKAN SERVER
# ==========================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=7860,
        debug=False
    )