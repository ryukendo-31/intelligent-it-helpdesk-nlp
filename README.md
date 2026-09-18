# Intelligent IT Helpdesk Ticket Classification and Resolution Assistant using NLP

## Overview

An end-to-end Text Analytics application for analyzing IT helpdesk
tickets. The system combines multilingual preprocessing, TF-IDF,
supervised classification, sentiment analysis, domain-specific entity
extraction, historical-ticket retrieval, and resolution suggestion.

## Problem Statement

IT helpdesks receive large numbers of natural-language tickets. Manually
categorizing tickets and searching previous cases for solutions is
time-consuming. This project uses NLP and machine learning to
automatically analyze a ticket, route it to a support queue, identify
useful information, and retrieve similar historical cases.

## Objectives

-   Preprocess multilingual IT support text.
-   Represent tickets using TF-IDF.
-   Compare Naive Bayes, SVM, and Decision Tree classifiers.
-   Predict the appropriate helpdesk queue.
-   Perform basic sentiment analysis.
-   Extract technical entities.
-   Retrieve similar historical tickets using cosine similarity.
-   Suggest a resolution using a previous support response.
-   Handle tickets with no relevant TF-IDF vocabulary.

## Dataset

**Customer IT Support -- Ticket Dataset (Kaggle)**

Source: https://www.kaggle.com/dsv/10323677

The working CSV contains **4,000 records and 17 columns**, including
`subject`, `body`, `answer`, `type`, `queue`, `priority`, `language`,
`business_type`, and `tag_1` to `tag_9`.

Languages observed include English, Spanish, German, French, and
Portuguese.

**Limitation:** the dataset is synthetic, so results should not be
interpreted as production performance on real enterprise helpdesk
traffic.

## Architecture

``` text
Incoming Ticket
      |
      v
Text Preprocessing
      |
      v
TF-IDF Representation
      |
      v
SVM Classification
      |
      v
Predicted Queue
      |
      v
Filter Historical Tickets
      |
      v
Cosine Similarity
      |
      v
Top Similar Tickets
      |
      v
Historical Support Answer
      |
      v
Suggested Resolution
```

Sentiment analysis and technical entity extraction are performed
alongside the main pipeline.

## Data Exploration

### Queue Distribution

  Queue                               Count
  --------------------------------- -------
  Technical Support                    1317
  Product Support                       690
  Customer Service                      627
  IT Support                            445
  Billing and Payments                  338
  Returns and Exchanges                 197
  Service Outages and Maintenance       141
  Sales and Pre-Sales                   137
  General Inquiry                        55
  Human Resources                        53

### Priority Distribution

  Priority     Count
  ---------- -------
  High          1649
  Medium        1603
  Low            748

The queue distribution is imbalanced, with Technical Support having the
most records and Human Resources and General Inquiry having the fewest.
This is why macro Precision, Recall, and F1 are reported alongside
accuracy.

The dataset contains four ticket types: Incident, Request, Change, and
Problem. No duplicate rows were observed during the initial exploration.

## Preprocessing

The preprocessing pipeline performs:

1.  Lowercasing
2.  Newline and HTML removal
3.  Punctuation removal
4.  Whitespace normalization
5.  Tokenization
6.  Language-specific stopword removal
7.  Language-specific Snowball stemming

Supported languages are English, Spanish, German, French, and
Portuguese.

## TF-IDF

The cleaned text is converted into numerical features using TF-IDF.

Configuration:

``` text
ngram_range = (1, 2)
min_df = 2
max_df = 0.95
sublinear_tf = True
```

Both unigrams and bigrams are used.

## Classification Results

An 80:20 stratified train-test split was used.

  Model             Accuracy   Macro Precision   Macro Recall   Macro F1
  --------------- ---------- ----------------- -------------- ----------
  Naive Bayes         36.88%            22.40%         13.30%     10.96%
  Linear SVM          58.00%            58.36%         50.58%     52.94%
  Decision Tree       41.88%            40.73%         41.52%     40.60%

The Linear SVM is used by the final analyzer because it produced the
highest measured values on this test split.

## Similar Ticket Retrieval

After classification, historical tickets are filtered to the predicted
queue. Tickets in the same language are preferred when available.

Similarity is calculated using TF-IDF cosine similarity:

``` text
Combined Similarity =
0.60 × Subject Similarity +
0.40 × Body Similarity
```

The top three historical tickets are returned.

A similarity threshold of **0.15** is used before a historical response
is accepted as a suggested resolution.

The similarity score is a retrieval score, not a probability or
confidence value.

## Sentiment Analysis

A lightweight lexicon-based analyzer counts positive and negative words:

``` text
Score = Positive Count - Negative Count
```

The result is Positive, Negative, or Neutral.

## Technical Entity Extraction

The final implementation uses domain-specific regular-expression rules
rather than a downloadable pretrained spaCy NER pipeline.

It recognizes:

-   Email addresses
-   IP addresses
-   Windows versions
-   macOS versions
-   Software versions
-   Software names
-   Device models
-   Device types
-   Network technologies

Example:

``` text
My Dell XPS 13 cannot boot after Windows 11 update.
```

Possible entities:

``` text
Dell XPS  -> DEVICE_MODEL
Windows 11 -> WINDOWS_VERSION
laptop -> DEVICE_TYPE
```

## Example End-to-End Test

Input:

``` text
Subject: Urgent laptop boot problem
Body: My Dell XPS 13 cannot boot after the Windows 11 update.
Language: en
```

Output:

``` text
Predicted Queue: IT Support
Sentiment: Negative
Sentiment Score: -2
```

The system extracted technical entities and retrieved a historical Dell
XPS boot issue with a similarity of approximately **42.73%**.

## Out-of-Domain Handling

If the processed ticket produces no known TF-IDF features, the system
returns `Out of Domain`.

The retrieval stage also rejects weak matches below the similarity
threshold rather than automatically presenting an unrelated historical
response.

## Project Structure

``` text
intelligent-it-helpdesk-nlp/
├── app/
├── data/
│   └── dataset-tickets-multi-lang3-4k.csv
├── models/
│   ├── tfidf_vectorizer.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   └── decision_tree.joblib
├── reports/
│   └── classification_results.csv
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── classification.py
│   ├── predict.py
│   ├── similarity.py
│   ├── sentiment.py
│   ├── ner.py
│   └── analyzer.py
├── architecture.md
├── requirements.txt
└── .gitignore
```

## Running the Project

Create and activate a virtual environment:

``` powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

``` powershell
pip install -r requirements.txt
```

Train models:

``` powershell
python src/classification.py
```

Run a prediction:

``` powershell
python src/predict.py
```

Run the complete analyzer:

``` powershell
python src/analyzer.py
```

The analyzer asks for a ticket subject, description, and language and
prints the queue, sentiment, entities, suggested resolution, and similar
historical tickets.

## Technologies

Python, Pandas, NumPy, Scikit-learn, NLTK, spaCy language resources,
Joblib, TF-IDF, cosine similarity, Multinomial Naive Bayes, Linear SVM,
Decision Tree, and regular expressions.

## Limitations

-   The dataset is synthetic.
-   Sentiment analysis is rule-based.
-   Entity extraction is rule-based.
-   TF-IDF retrieval depends mainly on vocabulary overlap.
-   The similarity threshold is heuristic.
-   The measured macro F1 of the selected classifier is 52.94%, so the
    system should be treated as an assistance tool rather than an
    autonomous routing system.

## Future Scope

Possible extensions include transformer embeddings, Sentence-BERT
retrieval, trained NER, transformer sentiment models, human feedback for
resolution ranking, stronger out-of-domain detection, and integration
with a real helpdesk platform.

## Conclusion

The project combines preprocessing, TF-IDF, supervised classification,
sentiment analysis, technical entity extraction, similarity-based
retrieval, and historical resolution suggestion into one practical IT
helpdesk NLP pipeline. The SVM achieved 58.00% accuracy and 52.94% macro
F1 on the test split.
