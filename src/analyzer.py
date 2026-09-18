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
from sentiment import analyze_sentiment
from ner import extract_entities


TFIDF_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.joblib"
)

SVM_PATH = os.path.join(
    MODEL_DIR,
    "svm.joblib"
)

SIMILARITY_THRESHOLD = 0.15


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

    return (
        df,
        tfidf,
        svm_model
    )


def calculate_similarity(
    subject,
    body,
    historical_df,
    tfidf
):

    historical_subjects = (
        historical_df["subject"]
        .fillna("")
        .astype(str)
    )

    historical_bodies = (
        historical_df["body"]
        .fillna("")
        .astype(str)
    )

    subject_vectors = tfidf.transform(
        historical_subjects
    )

    body_vectors = tfidf.transform(
        historical_bodies
    )

    new_subject_vector = tfidf.transform(
        [subject]
    )

    new_body_vector = tfidf.transform(
        [body]
    )

    subject_similarity = cosine_similarity(
        new_subject_vector,
        subject_vectors
    )[0]

    body_similarity = cosine_similarity(
        new_body_vector,
        body_vectors
    )[0]

    combined_similarity = (
        0.60 * subject_similarity
        +
        0.40 * body_similarity
    )

    return combined_similarity


def retrieve_similar_tickets(
    subject,
    body,
    language,
    predicted_queue,
    df,
    tfidf,
    top_k=3
):

    queue_df = df[
        df["queue"] == predicted_queue
    ].copy()

    language_df = queue_df[
        queue_df["language"] == language
    ].copy()

    if not language_df.empty:

        candidate_df = language_df

    else:

        candidate_df = queue_df

    if candidate_df.empty:

        return []

    similarities = calculate_similarity(
        subject,
        body,
        candidate_df,
        tfidf
    )

    candidate_df = candidate_df.copy()

    candidate_df["similarity"] = (
        similarities
    )

    candidate_df = candidate_df.sort_values(
        "similarity",
        ascending=False
    )

    candidate_df = candidate_df.head(
        top_k
    )

    results = []

    for _, row in candidate_df.iterrows():

        results.append({
            "similarity": float(
                row["similarity"]
            ),
            "subject": row["subject"],
            "body": row["body"],
            "queue": row["queue"],
            "priority": row["priority"],
            "type": row["type"],
            "language": row["language"],
            "answer": row["answer"]
        })

    return results


def analyze_ticket(
    subject,
    body,
    language,
    top_k=3
):

    (
        df,
        tfidf,
        svm_model
    ) = load_resources()

    full_text = (
        f"{subject} {body}"
    )

    sentiment_result = analyze_sentiment(
        full_text
    )

    entities = extract_entities(
        full_text
    )

    ticket_df = pd.DataFrame([{
        "subject": subject,
        "body": body,
        "answer": "",
        "language": language
    }])

    processed_ticket = prepare_dataset(
        ticket_df
    )

    if processed_ticket.empty:

        return {
            "queue": "Out of Domain",
            "sentiment": sentiment_result,
            "entities": entities,
            "similar_tickets": [],
            "suggested_resolution":
                "Unable to analyze the ticket."
        }

    clean_text = processed_ticket[
        "clean_text"
    ].iloc[0]

    vector = tfidf.transform(
        [clean_text]
    )

    if vector.nnz == 0:

        return {
            "queue": "Out of Domain",
            "sentiment": sentiment_result,
            "entities": entities,
            "similar_tickets": [],
            "suggested_resolution":
                "The ticket does not appear to "
                "contain terms related to the IT "
                "helpdesk dataset. Please enter an "
                "IT-related issue."
        }

    predicted_queue = svm_model.predict(
        vector
    )[0]

    similar_tickets = retrieve_similar_tickets(
        subject,
        body,
        language,
        predicted_queue,
        df,
        tfidf,
        top_k
    )

    suggested_resolution = ""

    if similar_tickets:

        best_similarity = (
            similar_tickets[0]["similarity"]
        )

        if best_similarity >= SIMILARITY_THRESHOLD:

            suggested_resolution = (
                similar_tickets[0]["answer"]
            )

        else:

            suggested_resolution = (
                "No sufficiently similar "
                "historical ticket was found. "
                "Please review the ticket manually."
            )

    else:

        suggested_resolution = (
            "No similar historical tickets found."
        )

    return {
        "queue": predicted_queue,
        "sentiment": sentiment_result,
        "entities": entities,
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
        "\nSentiment:",
        result["sentiment"]["sentiment"]
    )

    print(
        "Sentiment Score:",
        result["sentiment"]["score"]
    )

    print("\nNamed Entities:")

    if result["entities"]:

        for entity in result["entities"]:

            print(
                f"  {entity['text']} "
                f"-> {entity['label']}"
            )

    else:

        print(
            "  No entities found."
        )

    print(
        "\nSuggested Resolution:"
    )

    print(
        result["suggested_resolution"]
    )

    print("\n" + "=" * 80)
    print("SIMILAR HISTORICAL TICKETS")
    print("=" * 80)

    if not result["similar_tickets"]:

        print(
            "\nNo similar tickets found."
        )

    else:

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

            print(
                "Language:",
                ticket["language"]
            )
            