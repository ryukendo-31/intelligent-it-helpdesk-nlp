                 USER TICKET
                      │
                      ▼
             Text Preprocessing
                      │
                      ▼
                  TF-IDF
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
      Naive Bayes     SVM    Decision Tree
          │           │           │
          └───────────┼───────────┘
                      ▼
             Ticket Classification
                      │
                      ▼
              Similar Tickets
                      │
                      ▼
           Previous Agent Answers
                      │
                      ▼
          Suggested Resolution



intelligent-it-helpdesk-nlp/
│
├── app/
├── data/
│   └── dataset-tickets-multi-lang3-4k.csv
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── classification.py
│   ├── similarity.py
│   ├── sentiment.py
│   └── ner.py
│
├── architecture.md
├── requirements.txt
└── .gitignore


Data Exploration results
Queue:
This is class-imbalanced, especially the last two classes. That's important: we'll use macro F1, not just accuracy, when comparing our models.

Priority has 3 classes:
high      1649
medium    1603
low        748

And ticket type has 4 classes:
Incident
Request
Change
Problem