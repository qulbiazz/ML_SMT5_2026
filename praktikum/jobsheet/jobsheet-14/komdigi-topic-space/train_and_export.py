import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

APP_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    APP_DIR / "../../../../data/tabular/komdigi_hoaks.csv",
    APP_DIR / "../../../data/tabular/komdigi_hoaks.csv",
    APP_DIR / "data/tabular/komdigi_hoaks.csv",
]
DATA_PATH = next((path.resolve() for path in DATA_CANDIDATES if path.exists()), None)
if DATA_PATH is None:
    raise FileNotFoundError(
        "komdigi_hoaks.csv tidak ditemukan. Simpan file tersebut di folder data/tabular repositori."
    )

RANDOM_STATE = 42
data = pd.read_csv(DATA_PATH, usecols=["title", "body_text", "topics"])
data["text"] = data["title"].fillna("") + " " + data["body_text"].fillna("")
data["topics"] = data["topics"].fillna("").str.strip()
data = data.loc[data["text"].str.strip().ne("") & data["topics"].ne("")].copy()

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data["topics"])
class_ids = np.arange(len(label_encoder.classes_))

train_val_text, test_text, train_val_labels, test_labels = train_test_split(
    data["text"],
    labels,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=labels,
)
train_text, validation_text, train_labels, validation_labels = train_test_split(
    train_val_text,
    train_val_labels,
    test_size=0.125,
    random_state=RANDOM_STATE,
    stratify=train_val_labels,
)

vectorizer = TfidfVectorizer(max_features=512, ngram_range=(1, 2), min_df=2)
train_features = vectorizer.fit_transform(train_text).toarray().astype(np.float32)
validation_features = vectorizer.transform(validation_text).toarray().astype(np.float32)
test_features = vectorizer.transform(test_text).toarray().astype(np.float32)

weights = compute_class_weight(
    class_weight="balanced",
    classes=class_ids,
    y=train_labels,
)
class_weights = dict(zip(class_ids, weights))

# Load TensorFlow after vectorization to reduce peak memory during preprocessing.
import tensorflow as tf

tf.keras.utils.set_random_seed(RANDOM_STATE)
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(train_features.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(len(label_encoder.classes_), activation="softmax"),
    ]
)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_features,
    train_labels,
    validation_data=(validation_features, validation_labels),
    epochs=20,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ],
    verbose=1,
)

test_loss, test_accuracy = model.evaluate(test_features, test_labels, verbose=0)
predictions = np.argmax(model.predict(test_features, verbose=0), axis=1)
macro_f1 = f1_score(test_labels, predictions, average="macro", zero_division=0)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test macro-F1: {macro_f1:.4f}")
print(
    classification_report(
        test_labels,
        predictions,
        labels=class_ids,
        target_names=label_encoder.classes_,
        zero_division=0,
    )
)

model.save(APP_DIR / "komdigi_topic_model.h5")
with (APP_DIR / "komdigi_preprocessor.pkl").open("wb") as file:
    pickle.dump(
        {"vectorizer": vectorizer, "label_encoder": label_encoder},
        file,
    )
print(f"Model and preprocessor exported to {APP_DIR}")
