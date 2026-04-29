# TinySurgicalBERT — Complete Project Documentation

## What This Document Set Is

This is a full, self-contained explanation of every technical step in the TinySurgicalBERT project.
Each section covers one major concept from first principles through advanced implementation,
following the five-component teaching cycle: intuition → visualization → mathematics → numerical example → code.

---

## Complete Section List

| # | File | Topic | Key Concepts |
|---|------|--------|--------------|
| 1 | `01_problem_and_dataset.md` | The Problem and Dataset | OR scheduling, regression targets, feature inventory, duration distribution |
| 2 | `02_preprocessing.md` | Data Preprocessing | Leakage prevention, cyclic time encoding, imputation, categorical harmonisation |
| 3 | `03_text_encoding_bert.md` | Text Encoding and BERT | Word embeddings, BPE tokenisation, Transformer self-attention, domain adaptation |
| 4 | `04_knowledge_distillation.md` | Knowledge Distillation | Teacher-student training, distillation loss, INT8 quantisation, ONNX export |
| 5 | `05_feature_assembly_cv.md` | Feature Assembly and Cross-Validation | Vector concatenation, stratified k-fold, fold-wise preprocessing |
| 6 | `06_hyperparameter_optimisation.md` | Hyperparameter Optimisation with Optuna | Bayesian optimisation, TPE algorithm, search spaces, early stopping |
| 7 | `07_downstream_models.md` | The 8 Downstream Regression Models | Ridge, Lasso, ElasticNet, Random Forest, XGBoost, LightGBM, MLP — full objective functions |
| 8 | `08_evaluation_metrics.md` | Evaluation Metrics | MAE, MSE, RMSE, sMAPE, R² — full derivations and worked examples |
| 9 | `09_results_and_statistics.md` | Results and Statistical Comparison | Result tables, Wilcoxon signed-rank, FDR-BH correction, clinical interpretation |

---

## How to Read These Documents

Each section is self-contained — you do not need to read them in order, but the concepts build on each other.
Sections 3 and 4 are the most mathematically dense; read them carefully.
Section 7 is the broadest; it covers all eight models in one place.

**Every formula in every section follows this structure:**
1. Plain-language intuition before any symbols
2. LaTeX display formula
3. Symbol-by-symbol dictionary
4. Term-by-term breakdown explaining WHY each piece is there
5. Worked numerical example with real numbers
6. Interpretation of the result

---

## Project at a Glance

| Property | Value |
|---|---|
| Task | Surgical case duration regression (minutes) |
| Dataset | 180,370 cases, single institution |
| Pre-operative features | 24 structured + 1 free-text procedure name |
| Text encodings evaluated | Structured Only, SentenceBERT, Bio-ClinicalBERT, TinySurgicalBERT |
| Downstream models | 8 (Lin. Reg., Ridge, Lasso, ElasticNet, Rand. Forest, XGBoost, LightGBM, MLP) |
| Validation | 5-fold stratified cross-validation |
| Hyperparameter search | 20 Optuna TPE trials per model |
| Best result | TinySurgicalBERT + XGBoost: MAE = 26.38 ± 0.10 min, R² = 0.854 |
| TinySurgicalBERT size | 0.75 MB INT8 ONNX (614× smaller than teacher) |
| Inference latency | 0.64 ms per case on CPU |
