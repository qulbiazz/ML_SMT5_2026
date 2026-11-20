import csv
import os
from collections import Counter
from pathlib import Path

from flask import Flask, abort, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline

APP_DIR = Path(__file__).resolve().parent
MIN_PREDICTION_CONFIDENCE = 0.60
MIN_FEATURE_COVERAGE = 0.35


def find_training_dataset() -> Path:
    configured_path = os.environ.get("HOAX_TRAINING_CSV_PATH")
    candidates = ([Path(configured_path)] if configured_path else []) + [
        APP_DIR / "datasetUMPOHoax.csv",
        *(
            parent / "data" / "tabular" / "datasetUMPOHoax.csv"
            for parent in APP_DIR.parents
        ),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "datasetUMPOHoax.csv tidak ditemukan. Letakkan di folder data/tabular/ repository, "
        "di samping app.py, atau atur environment variable HOAX_TRAINING_CSV_PATH."
    )


def find_dataset() -> Path:
    configured_path = os.environ.get("KOMDIGI_CSV_PATH")
    candidates = ([Path(configured_path)] if configured_path else []) + [
        APP_DIR / "komdigi_hoaks.csv",
        *(
            parent / "data" / "tabular" / "komdigi_hoaks.csv"
            for parent in APP_DIR.parents
        ),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "komdigi_hoaks.csv tidak ditemukan. Letakkan di folder data/tabular/ repository, "
        "di samping app.py, atau atur environment variable KOMDIGI_CSV_PATH."
    )


def train_classifier(csv_path: Path):
    with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"tweet", "label"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Kolom wajib tidak ditemukan di {csv_path}: {', '.join(sorted(missing_columns))}"
            )

        samples = [
            (row.get("tweet", "").strip(), row.get("label", "").strip().lower())
            for row in reader
        ]

    samples = [
        (text, label)
        for text, label in samples
        if text and label in {"hoax", "valid"}
    ]
    texts = [text for text, _ in samples]
    labels = [label for _, label in samples]
    if len(set(labels)) != 2 or min(Counter(labels).values(), default=0) < 2:
        raise ValueError(
            f"Dataset {csv_path} harus memiliki minimal dua berita untuk masing-masing "
            "label 'hoax' dan 'valid'."
        )

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    model = make_pipeline(
        CountVectorizer(
            binary=True,
            max_features=20_000,
            min_df=2,
            ngram_range=(1, 2),
        ),
        BernoulliNB(alpha=1.0, fit_prior=False),
    )
    model.fit(train_texts, train_labels)
    test_predictions = model.predict(test_texts)
    metrics = {
        "accuracy": accuracy_score(test_labels, test_predictions),
        "macro_f1": f1_score(
            test_labels,
            test_predictions,
            average="macro",
            zero_division=0,
        ),
        "test_count": len(test_texts),
        "training_count": len(train_texts),
    }
    return model, metrics


def load_articles(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"id", "title", "category"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Kolom wajib tidak ditemukan di {csv_path}: {', '.join(sorted(missing_columns))}"
            )

        articles = []
        for row in reader:
            article = {key: (value or "").strip() for key, value in row.items() if key}
            article.setdefault("excerpt", "")
            article.setdefault("body_text", "")
            article.setdefault("published_at", "")
            article.setdefault("url", "")
            article.setdefault("view_count", "0")
            article.setdefault("tags", "")
            article.setdefault("topics", "")
            articles.append(article)

    return sorted(
        articles,
        key=lambda article: (article["published_at"], article["id"]),
        reverse=True,
    )


DATASET_PATH = find_dataset()
ARTICLES = load_articles(DATASET_PATH)
CATEGORIES = sorted({article["category"] for article in ARTICLES if article["category"]})
CATEGORY_COUNTS = Counter(article["category"] for article in ARTICLES)
TRAINING_DATASET_PATH = find_training_dataset()
CLASSIFIER, CLASSIFIER_METRICS = train_classifier(TRAINING_DATASET_PATH)

app = Flask(__name__)
app.config["ARTICLES_PER_PAGE"] = 20


@app.get("/")
def index():
    query = request.args.get("q", "").strip()
    selected_category = request.args.get("category", "").strip()
    try:
        page = max(1, int(request.args.get("page", "1")))
    except ValueError:
        page = 1

    filtered_articles = ARTICLES
    if selected_category in CATEGORIES:
        filtered_articles = [
            article for article in filtered_articles
            if article["category"] == selected_category
        ]
    elif selected_category:
        selected_category = ""

    if query:
        search_query = query.casefold()
        searchable_fields = ("title", "excerpt", "body_text", "tags", "topics")
        filtered_articles = [
            article for article in filtered_articles
            if any(search_query in article[field].casefold() for field in searchable_fields)
        ]

    per_page = app.config["ARTICLES_PER_PAGE"]
    total = len(filtered_articles)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = min(page, total_pages)
    start = (page - 1) * per_page

    return render_template(
        "index.html",
        articles=filtered_articles[start:start + per_page],
        categories=CATEGORIES,
        category_counts=CATEGORY_COUNTS,
        selected_category=selected_category,
        query=query,
        page=page,
        total_pages=total_pages,
        total=total,
        dataset_count=len(ARTICLES),
    )


@app.get("/hoaks/<article_id>")
def article_detail(article_id: str):
    article = next((item for item in ARTICLES if item["id"] == article_id), None)
    if article is None:
        abort(404)
    return render_template("detail.html", article=article)


@app.route("/klasifikasi", methods=["GET", "POST"])
def classify_news():
    prediction = None
    title = ""
    article_text = ""
    error = None
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        article_text = request.form.get("article_text", "").strip()
        combined_text = f"{title} {article_text}".strip()
        if not combined_text:
            error = "Masukkan judul atau isi berita terlebih dahulu."
        else:
            probabilities = CLASSIFIER.predict_proba([combined_text])[0]
            labels = CLASSIFIER.classes_
            predicted_index = probabilities.argmax()
            confidence = float(probabilities[predicted_index])
            vectorizer = CLASSIFIER.named_steps["countvectorizer"]
            analyzed_features = vectorizer.build_analyzer()(combined_text)
            feature_coverage = (
                sum(feature in vectorizer.vocabulary_ for feature in analyzed_features)
                / len(analyzed_features)
                if analyzed_features
                else 0.0
            )
            predicted_label = labels[predicted_index]
            use_valid_fallback = (
                predicted_label == "hoax"
                and (
                    confidence < MIN_PREDICTION_CONFIDENCE
                    or feature_coverage < MIN_FEATURE_COVERAGE
                )
            )
            prediction = {
                "label": "valid" if use_valid_fallback else predicted_label,
                "confidence": confidence,
                "used_valid_fallback": use_valid_fallback,
                "feature_coverage": feature_coverage,
            }

    return render_template(
        "classifier.html",
        title=title,
        article_text=article_text,
        prediction=prediction,
        error=error,
        metrics=CLASSIFIER_METRICS,
    ), (400 if error else 200)


if __name__ == "__main__":
    app.run(
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "").lower() == "true",
    )
