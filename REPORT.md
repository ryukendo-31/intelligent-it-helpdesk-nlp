# Intelligent IT Helpdesk Ticket Classification and Resolution Assistant using NLP

## Abstract

This project develops an NLP-based IT helpdesk analysis system that
automatically processes support tickets, predicts the appropriate
support queue, analyzes sentiment, extracts technical entities,
retrieves similar historical tickets, and suggests a possible resolution
from previous support responses.

The working dataset contains 4,000 IT support tickets. Multilingual
preprocessing is applied using language-specific stopword lists and
Snowball stemmers. TF-IDF with unigram and bigram features is used for
text representation. Multinomial Naive Bayes, Linear SVM, and Decision
Tree models are compared. Linear SVM achieved 58.00% accuracy and 52.94%
macro F1 on the test split and is used in the final analyzer.

## 1. Introduction

IT support tickets are naturally expressed in unstructured text and may
contain information about devices, software, errors, requests, and user
experience. A useful helpdesk assistant therefore needs more than a
simple classifier.

The developed system combines text preprocessing, classification,
information extraction, sentiment analysis, and retrieval into one
pipeline:

``` text
Ticket → Preprocessing → TF-IDF → Classification
       → Historical Retrieval → Suggested Resolution
```

## 2. Problem Statement

Manual ticket routing and searching for previous solutions can be
time-consuming when a helpdesk receives many natural-language tickets.
The project aims to automate the initial analysis of an IT ticket and
provide useful historical context to a support agent.

## 3. Objectives

1.  Preprocess multilingual support tickets.
2.  Convert text into TF-IDF features.
3.  Compare Naive Bayes, SVM, and Decision Tree classifiers.
4.  Predict the helpdesk queue.
5.  Perform sentiment analysis.
6.  Extract technical entities.
7.  Retrieve similar historical tickets.
8.  Suggest a resolution from historical responses.
9.  Provide basic out-of-domain protection.

## 4. Dataset

The project uses the **Customer IT Support -- Ticket Dataset** from
Kaggle.

Source: https://www.kaggle.com/dsv/10323677

The working CSV contains **4,000 records and 17 columns**.

Important fields are:

  Field            Description
  ---------------- ---------------------------------------
  subject          Ticket subject
  body             Ticket description
  answer           Historical support response
  type             Incident, Request, Change, or Problem
  queue            Support queue
  priority         Ticket priority
  language         Ticket language
  business_type    Business classification
  tag_1 to tag_9   Ticket tags

The observed languages include English, Spanish, German, French, and
Portuguese.

### Dataset Limitation

The dataset is synthetic. Consequently, the reported performance should
be interpreted as performance on this selected dataset and not as
production performance on real enterprise helpdesk traffic.

## 5. Data Exploration

### 5.1 Dataset Size

The dataset contains 4,000 records and 17 original columns.

No duplicate rows were observed during the initial exploration.

### 5.2 Queue Distribution

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

Technical Support is the largest class. General Inquiry and Human
Resources have substantially fewer examples.

This class imbalance is relevant when evaluating classifiers.
Macro-averaged metrics are therefore reported in addition to accuracy.

### 5.3 Priority Distribution

  Priority     Count
  ---------- -------
  High          1649
  Medium        1603
  Low            748

### 5.4 Ticket Types

Four ticket types are represented:

-   Incident
-   Request
-   Change
-   Problem

### 5.5 Language

The dataset is multilingual and contains English, Spanish, German,
French, and Portuguese tickets. This motivated the use of
language-specific preprocessing.

## 6. Methodology

The complete workflow is:

``` text
                    Incoming Ticket
                          |
                          v
                Text Preprocessing
                          |
                          v
                    TF-IDF Vector
                          |
                          v
                 SVM Queue Classifier
                          |
                          v
                  Predicted Queue
                          |
                          v
             Historical Ticket Filtering
                          |
                          v
                  Cosine Similarity
                          |
                          v
                Similar Historical Cases
                          |
                          v
                Historical Answer
                          |
                          v
                Suggested Resolution
```

Sentiment analysis and technical entity extraction operate on the
incoming ticket as additional analysis components.

## 7. Text Preprocessing

The following operations are applied:

1.  Lowercasing
2.  Newline removal
3.  HTML removal
4.  Punctuation removal
5.  Whitespace normalization
6.  Tokenization
7.  Language-specific stopword removal
8.  Language-specific Snowball stemming

Supported languages:

``` text
English
Spanish
German
French
Portuguese
```

Using language-specific preprocessing avoids applying an English
stopword list and stemmer indiscriminately to multilingual text.

## 8. TF-IDF Representation

The cleaned text is represented using Term Frequency-Inverse Document
Frequency.

Conceptually:

\[ TFIDF(t,d)=TF(t,d) imes IDF(t) \]

The implementation uses:

``` text
ngram_range = (1, 2)
min_df = 2
max_df = 0.95
sublinear_tf = True
```

Unigrams and bigrams allow the representation to capture both individual
technical terms and short phrases.

## 9. Classification

Three models were evaluated:

### 9.1 Multinomial Naive Bayes

A probabilistic classifier suitable for sparse text features.

### 9.2 Linear SVM

A linear Support Vector Machine trained on sparse TF-IDF vectors. Class
weighting is used to reduce the effect of class imbalance.

### 9.3 Decision Tree

A tree-based classifier with controlled maximum depth and class
weighting.

## 10. Experimental Setup

An 80:20 stratified train-test split was used.

The evaluation metrics were:

-   Accuracy
-   Macro Precision
-   Macro Recall
-   Macro F1-score

Macro metrics give equal importance to each class, which is useful when
classes have different numbers of examples.

## 11. Classification Results

  Model             Accuracy   Macro Precision   Macro Recall   Macro F1
  --------------- ---------- ----------------- -------------- ----------
  Naive Bayes         36.88%            22.40%         13.30%     10.96%
  SVM                 58.00%            58.36%         50.58%     52.94%
  Decision Tree       41.88%            40.73%         41.52%     40.60%

The Linear SVM produced the highest measured values among the three
models on this test split and was selected for the final application
pipeline.

The results also demonstrate the effect of model choice when all models
use the same TF-IDF representation and evaluation split.

## 12. Prediction

The final prediction module loads the trained TF-IDF vectorizer and SVM
model.

Example input:

``` text
Subject:
have trouble with speakers

Body:
the sound is very low cannot hear anything

Language:
en
```

Example result:

``` text
Predicted Queue: Product Support
```

## 13. Similar Historical Ticket Retrieval

Classification provides a queue, but it does not directly provide a
solution.

The retrieval process therefore:

1.  Predicts the queue.
2.  Filters historical tickets to that queue.
3.  Prefers tickets in the same language when available.
4.  Calculates cosine similarity.
5.  Returns the top three similar tickets.

Subject and body similarity are calculated separately:

\[ S = 0.60S\_{subject}+0.40S\_{body} \]

The subject receives greater weight because it normally provides a
concise description of the main issue.

## 14. Suggested Resolution

The answer associated with the highest-ranked historical ticket is used
as a suggested resolution when its similarity reaches the configured
threshold:

\[ Threshold = 0.15 \]

If no sufficiently similar ticket exists, the system asks for manual
review rather than automatically using a weak match.

The similarity value is a cosine-similarity retrieval score. It is not a
probability that the retrieved answer is correct.

## 15. Sentiment Analysis

A lightweight lexicon-based sentiment analyzer is implemented.

Positive and negative words are counted and the score is:

\[ Score = PositiveCount - NegativeCount \]

The output is:

``` text
Positive
Negative
Neutral
```

The analyzer includes domain-relevant words such as `error`, `broken`,
`unable`, `slow`, `frustrated`, `fixed`, `helpful`, and `satisfied`.

## 16. Entity Extraction

The final implementation uses domain-specific regular expressions rather
than a downloadable pretrained spaCy NER model.

Supported entities include:

  Entity             Example
  ------------------ -------------------------
  EMAIL              user@example.com
  IP_ADDRESS         192.168.1.10
  WINDOWS_VERSION    Windows 11
  MACOS_VERSION      macOS Ventura
  SOFTWARE_VERSION   Zoom 5.11.0
  SOFTWARE           Chrome, Teams, Outlook
  DEVICE_MODEL       Dell XPS
  DEVICE_TYPE        laptop, printer, router
  NETWORK_ENTITY     Wi-Fi, Bluetooth, VPN

This rule-based approach is particularly suitable for technical strings
with predictable patterns.

## 17. Integrated Analyzer

The `analyzer.py` module combines the main components:

``` text
Load dataset and models
        ↓
Sentiment analysis
        ↓
Entity extraction
        ↓
Preprocess ticket
        ↓
TF-IDF transformation
        ↓
Out-of-domain check
        ↓
SVM queue prediction
        ↓
Historical ticket retrieval
        ↓
Similarity threshold
        ↓
Suggested resolution
```

The returned result contains:

-   Predicted queue
-   Sentiment
-   Sentiment score
-   Named entities
-   Similar historical tickets
-   Similarity values
-   Suggested resolution

## 18. Test Case

Representative input:

``` text
Subject:
Urgent laptop boot problem

Body:
My Dell XPS 13 cannot boot after Windows 11 update.

Language:
en
```

Observed output:

``` text
Predicted Queue: IT Support
Sentiment: Negative
Sentiment Score: -2
```

Extracted entities included:

``` text
laptop -> DEVICE_TYPE
Dell XPS -> DEVICE_MODEL
Windows 11 -> WINDOWS_VERSION
Windows -> SOFTWARE
```

The top historical match was:

``` text
Urgent: Dell XPS 13 Boot Issue
```

with approximately **42.73% similarity**.

Other highly ranked results also concerned Dell XPS boot/startup issues,
demonstrating the intended connection between classification and
historical-ticket retrieval.

## 19. Out-of-Domain Handling

A TF-IDF vocabulary check is used before classification.

If the incoming ticket produces no known TF-IDF features, the system
returns:

``` text
Out of Domain
```

instead of blindly assigning a queue.

The similarity threshold provides an additional safeguard against using
a weak historical match as a resolution.

This is a basic safeguard rather than a complete semantic out-of-domain
detection method.

## 20. Software and Project Structure

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

Technologies:

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   NLTK
-   spaCy language resources
-   Joblib
-   Regular Expressions

## 21. Limitations

1.  The dataset is synthetic.
2.  The classification performance depends on the selected dataset and
    train-test split.
3.  Sentiment analysis is rule-based.
4.  Entity extraction is rule-based.
5.  TF-IDF similarity is mainly lexical and may miss semantic similarity
    expressed using different vocabulary.
6.  The 0.15 similarity threshold is heuristic and was not optimized
    using a separately labeled retrieval-validation dataset.
7.  The system is intended to assist support workflows rather than
    replace human support agents.

## 22. Future Scope

Possible improvements include:

-   Transformer-based text embeddings
-   Sentence-BERT semantic retrieval
-   Trained IT-specific NER
-   Transformer-based sentiment analysis
-   Human feedback for ranking resolutions
-   Better out-of-domain detection
-   Evaluation using manually labeled similar-ticket pairs
-   Integration with real helpdesk software

## 23. Conclusion

The project successfully combines several NLP techniques into a single
practical IT helpdesk analysis pipeline.

The system performs multilingual preprocessing, TF-IDF representation,
supervised classification, sentiment analysis, technical entity
extraction, historical-ticket retrieval, and resolution suggestion.

Among the evaluated models, Linear SVM achieved 58.00% accuracy and a
macro F1-score of 52.94% on the test split. The final analyzer uses this
model to classify tickets and then searches relevant historical cases
using queue-aware and language-aware cosine similarity.

The result is a lightweight, interpretable NLP-based assistant that
demonstrates the practical use of Text Analytics for IT helpdesk
automation.
