import re

from spacy.lang.en.stop_words import STOP_WORDS as EN_STOPWORDS
from spacy.lang.es.stop_words import STOP_WORDS as ES_STOPWORDS
from spacy.lang.de.stop_words import STOP_WORDS as DE_STOPWORDS
from spacy.lang.fr.stop_words import STOP_WORDS as FR_STOPWORDS
from spacy.lang.pt.stop_words import STOP_WORDS as PT_STOPWORDS

from nltk.stem.snowball import SnowballStemmer


STOPWORD_SETS = {
    "en": EN_STOPWORDS,
    "es": ES_STOPWORDS,
    "de": DE_STOPWORDS,
    "fr": FR_STOPWORDS,
    "pt": PT_STOPWORDS
}


STEMMERS = {
    "en": SnowballStemmer("english"),
    "es": SnowballStemmer("spanish"),
    "de": SnowballStemmer("german"),
    "fr": SnowballStemmer("french"),
    "pt": SnowballStemmer("portuguese")
}


def preprocess_text(text, language):
    text = str(text).lower()

    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\r+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()

    stopword_set = STOPWORD_SETS.get(language, set())

    tokens = [
        word for word in tokens
        if word not in stopword_set
    ]

    stemmer = STEMMERS.get(language)

    if stemmer:
        tokens = [
            stemmer.stem(word)
            for word in tokens
        ]

    return " ".join(tokens)


def prepare_dataset(df):
    df = df.copy()

    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df["answer"] = df["answer"].fillna("")

    df["text"] = (
        df["subject"].str.strip()
        + " "
        + df["body"].str.strip()
    ).str.strip()

    df = df[df["text"].str.len() > 0].copy()

    df["clean_text"] = df.apply(
        lambda row: preprocess_text(
            row["text"],
            row["language"]
        ),
        axis=1
    )

    df = df[df["clean_text"].str.len() > 0].copy()

    df.reset_index(drop=True, inplace=True)

    return df