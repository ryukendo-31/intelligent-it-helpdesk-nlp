import os
import sys
import joblib
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


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

SRC_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.insert(
    0,
    SRC_DIR
)

from preprocessing import prepare_dataset


TFIDF_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.joblib"
)

SVM_PATH = os.path.join(
    MODEL_DIR,
    "svm.joblib"
)


def load_resources():

    df = pd.read_csv(
        DATA_PATH
    )

    df = prepare_dataset(
        df
    )

    tfidf = joblib.load(
        TFIDF_PATH
    )

    svm_model = joblib.load(
        SVM_PATH
    )

    ticket_vectors = tfidf.transform(
        df["clean_text"]
    )

    return (
        df,
        tfidf,
        svm_model,
        ticket_vectors
    )


def find_similar_tickets(
    subject,
    body,
    language,
    top_k=3
):

    (
        df,
        tfidf,
        svm_model,
        ticket_vectors
    ) = load_resources()

    new_ticket = (
        f"{subject} {body}"
    )

    clean_ticket = prepare_dataset(
        pd.DataFrame([{
            "subject": subject,
            "body": body,
            "answer": "",
            "language": language
        }])
    )["clean_text"].iloc[0]

    new_vector = tfidf.transform(
        [clean_ticket]
    )

    predicted_queue = svm_model.predict(
        new_vector
    )[0]

    queue_mask = (
        df["queue"] == predicted_queue
    )

    filtered_df = df[
        queue_mask
    ].copy()

    filtered_vectors = ticket_vectors[
        queue_mask.values
    ]

    similarities = cosine_similarity(
        new_vector,
        filtered_vectors
    )[0]

    top_indices = similarities.argsort()[
        ::-1
    ][:top_k]

    results = []

    for index in top_indices:

        row = filtered_df.iloc[index]

        results.append({
            "similarity": similarities[index],
            "subject": row["subject"],
            "body": row["body"],
            "queue": row["queue"],
            "priority": row["priority"],
            "type": row["type"],
            "language": row["language"],
            "answer": row["answer"]
        })

    return predicted_queue, results


if __name__ == "__main__":

    subject = input(
        "Enter ticket subject: "
    )

    body = input(
        "Enter ticket description: "
    )

    language = input(
        "Enter language (en/es/de/fr/pt): "
    ).strip().lower()

    predicted_queue, results = find_similar_tickets(
        subject,
        body,
        language,
        top_k=3
    )

    print("\n" + "=" * 80)
    print("HELPDESK RESOLUTION ASSISTANT")
    print("=" * 80)

    print(
        "\nPredicted Queue:",
        predicted_queue
    )

    for i, result in enumerate(
        results,
        start=1
    ):

        print("\n" + "-" * 80)
        print(
            f"SIMILAR TICKET #{i}"
        )
        print("-" * 80)

        print(
            f"Similarity: "
            f"{result['similarity'] * 100:.2f}%"
        )

        print(
            "Subject:",
            result["subject"]
        )

        print(
            "Queue:",
            result["queue"]
        )

        print(
            "Priority:",
            result["priority"]
        )

        print(
            "Type:",
            result["type"]
        )

        print(
            "\nPrevious Agent Answer:"
        )

        print(
            result["answer"]
        )