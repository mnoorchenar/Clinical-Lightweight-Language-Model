# TinySurgicalBERT — Section 5: Feature Assembly and Cross-Validation

## Introduction

After preprocessing produces a 38-dimensional structured vector and text encoding produces
an embedding vector, we must combine them into a single input for the downstream regression model.
Then we must evaluate that model using a rigorous protocol that gives an honest estimate of
how it will perform on new patients.
This section covers both steps: **vector concatenation** and **stratified k-fold cross-validation**.

---

## 5.1 Feature Assembly: Vector Concatenation

### What It Is

Feature assembly is the act of combining two feature vectors into one longer vector
by placing them end to end — a mathematical operation called **concatenation**.

**Real-world analogy**: think of filling out a form about a patient.
The first half of the form has checkboxes and numbers (structured data: age, BMI, specialty).
The second half has a long text description converted into numbers (text embedding).
Concatenation staples these two halves together into one complete form
that the regression model reads as a single entity.

### Mathematical Formulation

For case $i$, the final feature vector is:

$$\mathbf{x}_i = \left[\mathbf{x}_i^{\text{struct}};\; \mathbf{e}_i^{\text{text}}\right] \in \mathbb{R}^{38 + d}$$

**Symbol definitions:**

- $\mathbf{x}_i^{\text{struct}} \in \mathbb{R}^{38}$: the 38-dimensional structured feature vector after one-hot encoding and cyclic temporal encoding
- $\mathbf{e}_i^{\text{text}} \in \mathbb{R}^{d}$: the text embedding; $d$ depends on encoder:
  - Structured Only: $d = 0$ (no text features, vector length = 38)
  - SentenceBERT: $d = 384$, total = 422
  - Bio-ClinicalBERT: $d = 768$, total = 806
  - TinySurgicalBERT: $d = 256$ (projected from 128 to 256), total = 294
- $[\mathbf{a};\mathbf{b}]$: concatenation operator — the notation for placing vector $\mathbf{a}$ followed by vector $\mathbf{b}$ as a single longer vector
- $38 + d$: the total number of features the downstream model receives

**Worked example for TinySurgicalBERT:**

$$\underbrace{\mathbf{x}_i^{\text{struct}}}_{38 \text{ dims}} \;=\; [0.1,\; 0.8,\; 1,\; 0,\; 0,\; \ldots]$$

$$\underbrace{\mathbf{e}_i^{\text{text}}}_{256 \text{ dims}} \;=\; [0.23,\; -0.11,\; 0.65,\; \ldots]$$

$$\mathbf{x}_i \;=\; [\underbrace{0.1,\; 0.8,\; 1,\; 0,\; 0,\; \ldots}_{\text{38 structured}},\; \underbrace{0.23,\; -0.11,\; 0.65,\; \ldots}_{\text{256 text}}] \in \mathbb{R}^{294}$$

This 294-dimensional vector is the single input $\mathbf{x}_i$ fed to XGBoost, Ridge,
LightGBM, etc.

### Visualisation of Concatenation

```{.matplotlib}
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)
struct = rng.normal(0, 1, 38)
text   = rng.normal(0, 1, 256)
full   = np.concatenate([struct, text])

fig, axes = plt.subplots(1, 3, figsize=(14, 3))
fig.patch.set_facecolor('#0A0A0A')
titles = ['Structured (38-d)', 'TinySurgBERT embedding (256-d)', 'Concatenated (294-d)']
vecs   = [struct, text, full]
colors = ['#F57C00', '#1565C0', '#2E7D32']

for ax, vec, title, color in zip(axes, vecs, titles, colors):
    ax.set_facecolor('#0A0A0A')
    for spine in ax.spines.values(): spine.set_color('#444444')
    ax.tick_params(colors='#CCCCCC', labelsize=10)
    ax.bar(np.arange(len(vec)), vec, color=color, width=1.0, edgecolor='none', alpha=0.85)
    ax.axhline(0, color='#888888', lw=0.8)
    ax.set_title(title, color='#CCCCCC', fontsize=11)
    ax.set_xlabel('Feature index', color='#CCCCCC')
    ax.set_ylabel('Value', color='#CCCCCC')

plt.tight_layout()
```

**What to observe**: The third panel (concatenated) is simply the first two panels placed
end-to-end.
The structured block (first 38 bars) shows mostly binary values (one-hot) mixed with a few
continuous values (age, BMI normalised).
The text block (remaining 256 bars) shows dense continuous values — typical of BERT embeddings
which fill their entire range with meaningful variation.

---

## 5.2 Five-Fold Stratified Cross-Validation

### What It Is

Cross-validation is the technique of assessing a model's performance by training and evaluating
it multiple times on different partitions of the data, then averaging the results.

**Real-world analogy**: suppose you are hiring a chef for a restaurant and want to know
how good they are.
You would not judge them on one meal they cooked at home.
Instead, you give them five different menus, each a fair mix of easy and hard dishes,
and score them on each.
Their average score across all five menus is a reliable estimate of their true ability.
Cross-validation does exactly this for machine learning models.

### Why 5-Fold?

With 180,370 cases and 5 folds:
- Each fold uses 144,296 cases for training and 36,074 for validation (80/20 split)
- Each case is in the validation set exactly once across all 5 runs
- Final metrics are computed as the mean and standard deviation across the 5 fold scores

Fewer folds (e.g., 3-fold) give less stable estimates; more folds (e.g., 10-fold) give
marginally better estimates but multiply computation cost.
5-fold is the standard practical choice.

### Why Stratification?

**Without stratification**: random shuffling might accidentally put all long surgeries
(300+ min) in fold 1, making that fold's validation set unrepresentative.
A model might look great on four easy folds and terrible on the one hard fold —
giving a misleading estimate of real-world performance.

**With stratification**: we bin the target variable (surgical duration) into $Q$ quantile
bins, then ensure each fold contains the same proportion of each bin.
Every fold sees a representative mix of short, medium, and long cases.

### Mathematical Formulation

**Step 1 — Quantile binning of target:**

$$q_i = \left\lfloor Q \cdot \frac{\text{rank}(y_i)}{N} \right\rfloor, \quad q_i \in \{0, 1, \ldots, Q-1\}$$

**Symbol definitions:**

- $y_i$: the duration of case $i$ in minutes
- $\text{rank}(y_i)$: the rank of case $i$ when all $N$ durations are sorted ascending (from 1 to $N$)
- $Q$: number of quantile bins (in this project, $Q = 10$)
- $\lfloor \cdot \rfloor$: floor function (round down to nearest integer)
- $q_i \in \{0, 1, \ldots, 9\}$: the quantile bin assignment for case $i$ — bin 0 is the shortest 10% of cases, bin 9 is the longest 10%

**Step 2 — Stratified fold assignment:**

Cases in each bin $q$ are randomly shuffled and then assigned cyclically to folds $\{0, 1, 2, 3, 4\}$.
This ensures approximately $N / (Q \times 5)$ cases from each quantile bin in each fold.

**Step 3 — Fold metrics:**

$$\text{MAE}_k = \frac{1}{|\mathcal{D}_{\text{val}}^{(k)}|} \sum_{i \in \mathcal{D}_{\text{val}}^{(k)}} |y_i - \hat{y}_i^{(k)}|$$

$$\overline{\text{MAE}} = \frac{1}{K} \sum_{k=0}^{K-1} \text{MAE}_k, \qquad \sigma_{\text{MAE}} = \sqrt{\frac{1}{K-1}\sum_{k=0}^{K-1}(\text{MAE}_k - \overline{\text{MAE}})^2}$$

**Symbol definitions:**

- $K = 5$: number of folds
- $\mathcal{D}_{\text{val}}^{(k)}$: the set of case indices in the validation partition of fold $k$
- $|\mathcal{D}_{\text{val}}^{(k)}|$: number of validation cases in fold $k$ (≈ 36,074)
- $\hat{y}_i^{(k)}$: the model's prediction for case $i$ when that case was in fold $k$'s validation set
- $\overline{\text{MAE}}$: mean MAE across all 5 folds — reported as the primary result
- $\sigma_{\text{MAE}}$: standard deviation across folds — reported as the ± value

### Numerical Example

Five fold MAE values for TinySurgicalBERT + XGBoost (from actual results):

| Fold | MAE (min) |
|---|---|
| 0 | 26.449 |
| 1 | 26.517 |
| 2 | 26.295 |
| 3 | 26.341 |
| 4 | 26.301 |

Mean:

$$\overline{\text{MAE}} = \frac{26.449 + 26.517 + 26.295 + 26.341 + 26.301}{5} = \frac{131.903}{5} = 26.381$$

Standard deviation:

Deviations from mean: $[+0.068,\; +0.136,\; -0.086,\; -0.040,\; -0.080]$

Squared deviations: $[0.00462,\; 0.01850,\; 0.00740,\; 0.00160,\; 0.00640]$

$$\sigma_{\text{MAE}} = \sqrt{\frac{0.00462 + 0.01850 + 0.00740 + 0.00160 + 0.00640}{4}} = \sqrt{\frac{0.03852}{4}} = \sqrt{0.00963} = 0.098$$

**Reported result**: MAE = $26.38 \pm 0.10$ minutes.

**Interpretation**: The ± 0.10 is very small relative to the mean of 26.38 — the model is
highly consistent across folds.
This stability confirms that the performance is not an artefact of lucky data splits;
the model generalises reliably across all patient subgroups.

---

## 5.3 Full Pipeline Overview

```{.graphviz}
digraph CV {
    graph [fontsize=20, dpi=150, size="9,9", ratio=auto,
           margin=0.2, nodesep=0.5, ranksep=0.45,
           fontname="DejaVu Sans", bgcolor="transparent"];
    node  [shape=box, style="rounded,filled", fontsize=17,
           fontname="DejaVu Sans", fontcolor=white, margin=0.18];
    edge  [fontsize=15, penwidth=2, arrowsize=1.2,
           color="#F57C00", fontname="DejaVu Sans"];
    rankdir=TB;

    data [label="Full dataset\n180,370 × 294 features",
          fillcolor="#1976D2", color="#0D3B6E"];

    subgraph cluster_split {
        style=filled; fillcolor="#1B3A1B"; fontcolor=white;
        fontname="DejaVu Sans"; fontsize=17; label="Stratified 5-fold split";
        f0 [label="Fold 0\n144K train / 36K val", fillcolor="#388E3C"];
        f1 [label="Fold 1\n144K train / 36K val", fillcolor="#388E3C"];
        f2 [label="Fold 2", fillcolor="#388E3C"];
        f3 [label="Fold 3", fillcolor="#388E3C"];
        f4 [label="Fold 4\n144K train / 36K val", fillcolor="#388E3C"];
    }
    subgraph cluster_fold {
        style=filled; fillcolor="#3E0A6E"; fontcolor=white;
        fontname="DejaVu Sans"; fontsize=17; label="Per-fold operations (for each fold k)";
        impute [label="Fold-wise imputation\n(medians from train only)", fillcolor="#7B1FA2"];
        ohe    [label="One-hot encoding\n(fitted on train)", fillcolor="#7B1FA2"];
        train  [label="Train model\n(Optuna HPO → best params)", fillcolor="#7B1FA2"];
        pred   [label="Predict on validation set", fillcolor="#7B1FA2"];
        metric [label="Compute MAE, R², RMSE,\nsMAPE for this fold", fillcolor="#7B1FA2"];
    }
    agg [label="Aggregate: mean ± std\nacross 5 folds\n→ Final reported metrics",
         fillcolor="#BF360C", color="#5C1A00"];

    data -> f0; data -> f1; data -> f2; data -> f3; data -> f4;
    f0 -> impute; f1 -> impute; f2 -> impute; f3 -> impute; f4 -> impute;
    impute -> ohe -> train -> pred -> metric -> agg;
}
```

**What to observe**: Every fold independently runs the full preprocessing-train-evaluate cycle.
The imputation medians and one-hot encoder are fitted fresh for each fold's training data —
this is the key safeguard against leakage from the validation set into training.

---

## 5.4 Code: Feature Assembly and Cross-Validation

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# --- Feature Assembly ---
def assemble_features(df_struct, embeddings):
    """
    df_struct : pd.DataFrame of structured features (N x 38 after preprocessing)
    embeddings: np.ndarray of text embeddings (N x d)
    Returns   : np.ndarray (N x (38+d))
    """
    X_struct = df_struct.values.astype(np.float32)
    X_text   = embeddings.astype(np.float32)
    return np.concatenate([X_struct, X_text], axis=1)  # column-wise join

# X.shape = (180370, 294) for TinySurgicalBERT

# --- Stratified 5-fold split ---
N_FOLDS   = 5
N_QUANTILES = 10

y = df['duration_min'].values    # target variable
quantile_bins = pd.qcut(y, q=N_QUANTILES, labels=False, duplicates='drop')

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
# StratifiedKFold.split(X, y) requires a classification label;
# we use quantile_bins as a proxy categorical target for stratification

fold_results = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, quantile_bins)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Fold-wise imputation: compute median on training, apply to both
    train_medians = np.nanmedian(X_train, axis=0)    # shape: (294,)
    nan_train = np.where(np.isnan(X_train))
    nan_val   = np.where(np.isnan(X_val))
    X_train[nan_train] = train_medians[nan_train[1]]  # fill NaN in train
    X_val[nan_val]     = train_medians[nan_val[1]]    # fill NaN in val using TRAIN median

    print(f"Fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}")
    # Output: Fold 0: train=144296, val=36074
    # Output: Fold 1: train=144296, val=36074
    # Output: ...

    # ... (model training and evaluation happens here — see Section 7) ...
    fold_results.append({'fold': fold_idx, 'val_size': len(val_idx)})

# Aggregate
print(f"\nMean val size across folds: {np.mean([r['val_size'] for r in fold_results]):.0f}")
# Output: Mean val size across folds: 36074
```

---

## Summary

| Concept | Key Takeaway |
|---|---|
| Feature concatenation | Structured (38-d) + text (d-d) → single $(38+d)$-d vector |
| TinySurgicalBERT total dims | 38 + 256 = 294 features |
| Stratified k-fold | Each fold gets a proportional share of short/long cases |
| Why stratify? | Prevents accidentally easy/hard fold splits that distort estimates |
| Fold-wise imputation | Medians computed on training set only — prevents val-to-train leakage |
| Final metric format | $\overline{\text{MAE}} \pm \sigma_{\text{MAE}}$ across 5 folds |
| MAE for best model | $26.38 \pm 0.10$ minutes |
