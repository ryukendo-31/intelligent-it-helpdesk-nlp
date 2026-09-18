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

sys.path.insert(0, SRC_DIR)

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

    df = pd.read_csv(DATA_PATH)

    df = prepare_dataset(df)

    tfidf = joblib.load(TFIDF_PATH)

    svm_model = joblib.load(SVM_PATH)

    ticket_vectors = tfidf.transform(
        df["clean_text"]
    )

    return (
        df,
        tfidf,
        svm_model,
        ticket_vectors
    )


def analyze_ticket(
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

    ticket_df = pd.DataFrame([{
        "subject": subject,
        "body": body,
        "answer": "",
        "language": language
    }])

    processed_ticket = prepare_dataset(
        ticket_df
    )

    clean_text = processed_ticket[
        "clean_text"
    ].iloc[0]

    vector = tfidf.transform(
        [clean_text]
    )

    predicted_queue = svm_model.predict(
        vector
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
        vector,
        filtered_vectors
    )[0]

    top_indices = similarities.argsort()[
        ::-1
    ][:top_k]

    similar_tickets = []

    for index in top_indices:

        row = filtered_df.iloc[index]

        similar_tickets.append({
            "similarity": float(
                similarities[index]
            ),
            "subject": row["subject"],
            "body": row["body"],
            "queue": row["queue"],
            "priority": row["priority"],
            "type": row["type"],
            "language": row["language"],
            "answer": row["answer"]
        })

    suggested_resolution = ""

    if similar_tickets:

        suggested_resolution = (
            similar_tickets[0]["answer"]
        )

    return {
        "queue": predicted_queue,
        "similar_tickets": similar_tickets,
        "suggested_resolution":
            suggested_resolution
    }


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

    result = analyze_ticket(
        subject,
        body,
        language
    )

    print("\n" + "=" * 80)
    print("INTELLIGENT IT HELPDESK ANALYSIS")
    print("=" * 80)

    print(
        "\nPredicted Queue:",
        result["queue"]
    )

    print(
        "\nSuggested Resolution:"
    )

    print(
        result["suggested_resolution"]
    )

    for i, ticket in enumerate(
        result["similar_tickets"],
        start=1
    ):

        print("\n" + "-" * 80)

        print(
            f"Similar Ticket #{i}"
        )

        print(
            f"Similarity: "
            f"{ticket['similarity'] * 100:.2f}%"
        )

        print(
            "Subject:",
            ticket["subject"]
        )

        print(
            "Priority:",
            ticket["priority"]
        )

        print(
            "Type:",
            ticket["type"]
        )