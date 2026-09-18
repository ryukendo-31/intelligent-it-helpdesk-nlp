import re


POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "helpful",
    "thanks",
    "thank",
    "working",
    "fixed",
    "solved",
    "easy",
    "happy",
    "satisfied",
    "fast",
    "perfect"
}


NEGATIVE_WORDS = {
    "bad",
    "problem",
    "issue",
    "error",
    "fail",
    "failed",
    "failure",
    "broken",
    "not",
    "unable",
    "cannot",
    "can't",
    "slow",
    "angry",
    "frustrated",
    "terrible",
    "poor",
    "wrong",
    "crash",
    "crashed",
    "lost",
    "blocked"
}


def analyze_sentiment(text):

    text = str(text).lower()

    words = re.findall(
        r"\b[\w']+\b",
        text
    )

    positive_count = sum(
        1 for word in words
        if word in POSITIVE_WORDS
    )

    negative_count = sum(
        1 for word in words
        if word in NEGATIVE_WORDS
    )

    score = (
        positive_count
        - negative_count
    )

    if score > 0:
        sentiment = "Positive"

    elif score < 0:
        sentiment = "Negative"

    else:
        sentiment = "Neutral"

    return {
        "sentiment": sentiment,
        "score": score,
        "positive_words": positive_count,
        "negative_words": negative_count
    }


if __name__ == "__main__":

    text = input(
        "Enter ticket text: "
    )

    result = analyze_sentiment(text)

    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS")
    print("=" * 60)

    print(
        "Sentiment:",
        result["sentiment"]
    )

    print(
        "Score:",
        result["score"]
    )

    print(
        "Positive words:",
        result["positive_words"]
    )

    print(
        "Negative words:",
        result["negative_words"]
    )