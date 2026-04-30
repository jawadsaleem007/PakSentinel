# PakSentinel — MLFlow Experiment Hierarchy Diagram

**Submitted before Task 6 implementation as required by rubric.**

---

## Experiment Hierarchy

```
MLFlow Tracking Server (file:///mlruns)
│
├── Experiment: PakSentinel_Preprocessing_Ablation
│   │   Purpose: Determine optimal preprocessing configuration
│   │   Varying: stopword list, normalization, min token length, max features
│   │
│   ├── Run: config_1
│   │   Params: standard stopwords, lemma, len≥1, 5K features
│   │   Metrics: accuracy, precision, recall, F1-weighted, ROC-AUC, time
│   │   Artifacts: confusion_matrix.png, roc_curve.png, vocabulary.json, report.txt
│   │
│   ├── Run: config_2
│   │   Params: custom stopwords, lemma, len≥1, 5K features
│   │
│   ├── Run: config_3
│   │   Params: custom stopwords, stem, len≥1, 5K features
│   │
│   ├── Run: config_4
│   │   Params: custom stopwords, lemma, len≥3, 5K features
│   │
│   ├── Run: config_5
│   │   Params: custom stopwords, lemma, len≥1, 10K features
│   │
│   └── Run: config_6
│       Params: custom stopwords, lemma, len≥1, 15K features
│
├── Experiment: PakSentinel_Feature_Comparison
│   │   Purpose: Compare feature representation methods
│   │
│   ├── Run: TF-IDF Only (dim=10000)
│   ├── Run: Word2Vec Only (dim=200)
│   └── Run: TF-IDF + Word2Vec (dim=10200)
│
├── Experiment: PakSentinel_Model_Comparison
│   │   Purpose: Compare classifier architectures
│   │
│   ├── Run: Naive Bayes (from scratch, BoW features)
│   ├── Run: Logistic Regression L1 (TF-IDF, C=1.0)
│   ├── Run: Logistic Regression L2 (TF-IDF, C=1.0)
│   ├── Run: Logistic Regression ElasticNet (TF-IDF, C=1.0)
│   ├── Run: Polynomial LR degree=1 (PCA 2D)
│   ├── Run: Polynomial LR degree=2 (PCA 2D)
│   └── Run: Polynomial LR degree=3 (PCA 2D)
│
└── Model Registry
    ├── PakSentinel_NB
    │   └── Promotion: Staging → Production if F1 improves by ≥ 1%
    ├── PakSentinel_LR
    │   └── Promotion: Staging → Production if F1 improves by ≥ 1%
    └── PakSentinel_PolyLR
        └── Promotion: Staging → Production if F1 improves by ≥ 1%
```

---

## Parameters Logged Per Run

| Parameter | Description | Example |
|-----------|-------------|---------|
| dataset_sources | Comma-separated source names | LIAR,FakeNewsNet,SarcasmHeadlines |
| train_size | Number of training samples | 7912 |
| test_size | Number of test samples | 1979 |
| tokenizer | Tokenizer used | nltk_word_tokenize |
| stopword_list | Stopword configuration | custom_199 |
| normalization | Normalization method | wordnet_lemmatizer |
| vectorizer_type | Feature extraction method | tfidf_sublinear |
| max_features | Maximum vocabulary size | 10000 |
| model_type | Classifier name | LogisticRegression_L2 |

## Metrics Logged Per Run

| Metric | Description |
|--------|-------------|
| accuracy | Overall classification accuracy |
| precision_fake | Precision for Fake class |
| recall_fake | Recall for Fake class |
| f1_fake | F1-score for Fake class |
| precision_real | Precision for Real class |
| recall_real | Recall for Real class |
| f1_real | F1-score for Real class |
| precision_satire | Precision for Satire class |
| recall_satire | Recall for Satire class |
| f1_satire | F1-score for Satire class |
| f1_weighted | Weighted F1-score (primary metric) |
| roc_auc | ROC Area Under Curve |
| training_time_s | Training time in seconds |

## Artifacts Per Run

| Artifact | Format | Description |
|----------|--------|-------------|
| confusion_matrix | PNG | Per-class confusion matrix heatmap |
| roc_curve | PNG | ROC curve with AUC annotations |
| vocabulary | JSON | Top 100 TF-IDF features |
| classification_report | TXT | Full sklearn classification report |
