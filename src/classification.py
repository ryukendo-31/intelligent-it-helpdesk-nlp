import os
import sys
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "dataset-tickets-multi-lang3-4k.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

sys.path.insert(
    0,
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from preprocessing import prepare_dataset


def load_data():
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    return df


def train_models(df):

    X = df["clean_text"]
    y = df["queue"]

    print("\n" + "=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples :", len(X_test))

    print("\n" + "=" * 70)
    print("TF-IDF")
    print("=" * 70)

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print("Training TF-IDF shape:", X_train_tfidf.shape)
    print("Testing TF-IDF shape :", X_test_tfidf.shape)
    print("Vocabulary size:", len(tfidf.vocabulary_))

    models = {
        "Naive Bayes": MultinomialNB(),

        "SVM": LinearSVC(
            class_weight="balanced",
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
            max_depth=100
        )
    }

    results = []
    trained_models = {}

    for name, model in models.items():

        print("\n" + "=" * 70)
        print("TRAINING:", name)
        print("=" * 70)

        model.fit(
            X_train_tfidf,
            y_train
        )

        predictions = model.predict(
            X_test_tfidf
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        trained_models[name] = model

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )

        print("\nConfusion Matrix:")
        print(
            confusion_matrix(
                y_test,
                predictions
            )
        )

    results_df = pd.DataFrame(results)

    return (
        tfidf,
        trained_models,
        results_df
    )


def save_models(tfidf, trained_models):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    joblib.dump(
        tfidf,
        os.path.join(
            MODEL_DIR,
            "tfidf_vectorizer.joblib"
        )
    )

    for name, model in trained_models.items():

        filename = (
            name.lower()
            .replace(" ", "_")
            + ".joblib"
        )

        joblib.dump(
            model,
            os.path.join(
                MODEL_DIR,
                filename
            )
        )

    print("\n" + "=" * 70)
    print("MODELS SAVED")
    print("=" * 70)

    print("Saved to:", MODEL_DIR)

    print("tfidf_vectorizer.joblib")
    print("naive_bayes.joblib")
    print("svm.joblib")
    print("decision_tree.joblib")


def main():

    df = load_data()

    print("\n" + "=" * 70)
    print("TEXT PREPROCESSING")
    print("=" * 70)

    df = prepare_dataset(df)

    print("Processed dataset shape:", df.shape)

    print("\nSample preprocessing:")

    for i in range(min(3, len(df))):

        print("\n" + "-" * 70)

        print("Language:")
        print(df.iloc[i]["language"])

        print("\nOriginal:")
        print(df.iloc[i]["text"])

        print("\nProcessed:")
        print(df.iloc[i]["clean_text"])

    tfidf, trained_models, results_df = train_models(df)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    save_models(
        tfidf,
        trained_models
    )

    results_path = os.path.join(
        BASE_DIR,
        "reports",
        "classification_results.csv"
    )

    os.makedirs(
        os.path.dirname(results_path),
        exist_ok=True
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print("\nResults saved to:")
    print(results_path)


if __name__ == "__main__":
    main()