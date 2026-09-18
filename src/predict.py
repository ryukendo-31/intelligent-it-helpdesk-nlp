import os
import sys
import joblib

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
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

from preprocessing import preprocess_text


TFIDF_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.joblib"
)

SVM_PATH = os.path.join(
    MODEL_DIR,
    "svm.joblib"
)


tfidf = joblib.load(TFIDF_PATH)
svm_model = joblib.load(SVM_PATH)


def predict_ticket(subject, body, language):

    text = f"{subject} {body}"

    clean_text = preprocess_text(
        text,
        language
    )

    vectorized_text = tfidf.transform(
        [clean_text]
    )

    prediction = svm_model.predict(
        vectorized_text
    )[0]

    return prediction


if __name__ == "__main__":

    subject = input("Enter ticket subject: ")
    body = input("Enter ticket description: ")
    language = input(
        "Enter language (en/es/de/fr/pt): "
    ).strip().lower()

    prediction = predict_ticket(
        subject,
        body,
        language
    )

    print("\n" + "=" * 60)
    print("HELPDESK TICKET ANALYSIS")
    print("=" * 60)

    print("Subject:", subject)
    print("Language:", language)
    print("Predicted Queue:", prediction)