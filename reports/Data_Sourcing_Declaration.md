# PakSentinel — Data Sourcing Declaration & Reliability Scorecards

**Course:** Natural Language Processing  
**Assignment:** 2 — PakSentinel  
**Deliverable:** Week 1 — Data Sourcing Declaration  
**Date:** April 2026

---

## 1. Data Sources Declaration

We hereby declare the following three datasets were used to construct the PakSentinel misinformation detection corpus:

| # | Source | Type | Download Method | License | Samples Used |
|---|--------|------|-----------------|---------|-------------|
| 1 | LIAR Dataset (Wang, 2017) | Political statements | Direct download (TSV) | Research use | 2,999 |
| 2 | FakeNewsNet (Shu et al., 2020) | News article titles | GitHub repository (CSV) | MIT License | 3,910 |
| 3 | Sarcasm Headlines (Misra & Grover, 2021) | Satirical headlines | Kaggle (JSON) | CC0 Public Domain | 2,995 |

**Total dataset size:** 9,904 samples  
**Classes:** Real (3,480 / 35.1%), Fake (3,429 / 34.6%), Satire (2,995 / 30.2%)  
**Duplicate rate:** 0.00% (after deduplication)

---

## 2. Data Reliability Scorecards

### 2.1 LIAR Dataset — Score: 18/25

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Label Credibility** | 5/5 | Labels from PolitiFact, a Pulitzer Prize-winning fact-checking organization. Human-annotated with 6-point scale (pants-fire, false, barely-true, half-true, mostly-true, true), mapped to binary Real/Fake. |
| **Recency** | 3/5 | Covers statements from 2007–2016. While political misinformation patterns remain relevant, the temporal gap means some topic-specific patterns may be outdated. |
| **Domain Relevance to Pakistan** | 2/5 | Primarily US political statements. Linguistic patterns of misinformation (hedging, exaggeration, emotional language) are cross-cultural, but topic distribution is US-centric. |
| **Class Balance** | 4/5 | Roughly balanced across 6 labels. After binary mapping, balance is approximately 55/45 (Real/Fake). Acceptable for training without correction. |
| **Language Consistency** | 4/5 | Consistent English political statements. Short-form text (1–2 sentences). Minimal noise, no code-switching. Suitable for headline-level classification. |

### 2.2 FakeNewsNet — Score: 17/25

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Label Credibility** | 5/5 | Labels derived from PolitiFact (fact-checking) and GossipCop (entertainment) professional fact-checkers. Binary Real/Fake labels with high inter-annotator reliability. |
| **Recency** | 3/5 | Covers 2015–2018 news articles. Captures political misinformation patterns from US election cycles. Entertainment fake news patterns remain stable over time. |
| **Domain Relevance to Pakistan** | 2/5 | US-centric news articles. However, stylistic differences between real and fake news (sensationalism, emotional manipulation) are language-universal features applicable across cultures. |
| **Class Balance** | 3/5 | PolitiFact subset: ~400 fake, ~400 real. GossipCop: ~5,000 fake, ~16,000 real. Combined imbalance addressed via undersampling during dataset construction. |
| **Language Consistency** | 4/5 | News article titles from professional sources. Consistent formatting across PolitiFact and GossipCop subsets. Standard English with minimal noise. |

### 2.3 Sarcasm Headlines Dataset — Score: 18/25

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **Label Credibility** | 5/5 | Labels are inherently reliable: The Onion is a well-known satirical publication, and HuffPost is a mainstream news outlet. Source-based labeling eliminates annotator subjectivity. |
| **Recency** | 4/5 | Headlines span 2012–2018, covering a broad temporal range. Satire patterns (irony, absurdity, exaggeration) are temporally stable linguistic constructs. |
| **Domain Relevance to Pakistan** | 2/5 | US-centric satirical content. However, the assignment requires a Satire class and The Onion represents the gold standard of English-language satire for NLP research. |
| **Class Balance** | 3/5 | Contains approximately 11,700 sarcastic and 14,900 non-sarcastic headlines. We use only the sarcastic subset for the Satire class, undersampled to match other classes. |
| **Language Consistency** | 4/5 | Consistent headline format throughout. The Onion maintains high editorial quality for satire. Short-form text compatible with other dataset sources. |

---

## 3. Source Combination Justification (324 words)

Our dataset construction strategy combines three complementary sources — LIAR, FakeNewsNet, and the Sarcasm Headlines Dataset — to create a robust three-class (Real, Fake, Satire) corpus for misinformation detection. This combination is motivated by both practical and theoretical considerations grounded in NLP literature on dataset bias and misinformation detection.

The LIAR dataset (Wang, 2017) provides fine-grained credibility labels from PolitiFact, a Pulitzer Prize-winning fact-checking organization. Its 6-level annotation scheme (pants-fire to true) allows flexible binary mapping while preserving label reliability. Research by Rashkin et al. (2017) demonstrated that linguistic features from political fact-checking datasets generalize well to broader misinformation contexts, making LIAR an excellent foundation despite its US-centric focus.

The FakeNewsNet dataset (Shu et al., 2020) contributes news article titles from two domains: PolitiFact (political fact-checking) and GossipCop (entertainment news verification). This multi-domain coverage enables our pipeline to learn features that generalize beyond a single topic area. Pérez-Rosas et al. (2018) showed that stylometric and lexical features of fake news (sensationalism, emotional manipulation, lack of attribution) are language-universal, supporting cross-domain applicability.

The Sarcasm Headlines Dataset (Misra & Grover, 2021) fills the critical Satire class using The Onion's satirical headlines. Satire detection is a known challenge in misinformation research because satire mimics fake news structure while serving a different communicative purpose (Rubin et al., 2016). Including this class prevents the common pitfall of binary classifiers that conflate satire with misinformation, a significant source of false positives in production systems.

Regarding dataset bias, we acknowledge the Western-centric nature of all three sources. Augenstein et al. (2019) demonstrated that cross-cultural bias in training data can reduce model performance on underrepresented populations. To mitigate this, we focus on language-universal features (TF-IDF patterns, stylometric markers, sentiment distributions) rather than topic-specific or culturally-bound features. Our preprocessing pipeline includes Roman Urdu handling to prepare for future Pakistan-specific data integration.

We apply undersampling to address class imbalance, following He & Garcia (2009) that undersampling is preferred over oversampling when sufficient data exists in all classes, avoiding SMOTE overfitting risks on text data (Blagus & Lusa, 2013).

---

## 4. References

1. Augenstein, I., et al. (2019). "MultiFC: A real-world multi-domain dataset for evidence-based fact checking of claims." EMNLP.
2. Blagus, R., & Lusa, L. (2013). "SMOTE for high-dimensional class-imbalanced data." BMC Bioinformatics, 14(1), 106.
3. He, H., & Garcia, E. A. (2009). "Learning from imbalanced data." IEEE TKDE, 21(9).
4. Misra, R., & Grover, J. (2021). "Sculpting Data for ML: The first act of Machine Learning."
5. Pérez-Rosas, V., et al. (2018). "Automatic detection of fake news." COLING.
6. Rashkin, H., et al. (2017). "Truth of varying shades." EMNLP.
7. Rubin, V. L., et al. (2016). "Fake news or truth? Using satirical cues to detect potentially misleading news." NAACL Workshop.
8. Shu, K., et al. (2020). "FakeNewsNet: A data repository." Big Data, 8(3).
9. Wang, W. Y. (2017). "Liar, liar pants on fire." ACL.
