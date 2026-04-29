# TinySurgicalBERT — Section 1: The Problem and Dataset

## Introduction

Before writing a single line of code, every machine learning project begins with one question:
**what exactly are we trying to predict, and what information do we have to predict it with?**
This section answers both questions in full.
We will cover the clinical problem, the structure of the dataset, the target variable, and the
statistical properties of surgical duration — because understanding the shape of your data
determines every modelling decision that follows.

---

## 1.1 The Clinical Problem: Why Predict Surgical Duration?

### What It Is

Surgical case duration prediction means: given information available **before** a patient enters
the operating room, produce a number representing how many minutes the surgery will take.

This is a **regression problem** — the output is a continuous number, not a category.
The model is essentially a very informed estimator sitting at the scheduling desk every morning,
looking at the day's case list and writing a time estimate next to each patient's name.

**Real-world analogy**: think of a GPS navigator estimating your travel time before you leave.
It does not know about the traffic jam you will hit in 20 minutes, but it knows your starting
point, destination, road type, time of day, and historical average speeds — and it produces a
good estimate from those pre-departure features.
Our model works the same way: it does not know that the surgeon will encounter unexpected
bleeding at minute 45, but it knows the procedure name, surgeon ID, patient demographics,
and historical averages — and produces a pre-operative estimate.

### Why This Problem Exists

Operating rooms are the most expensive rooms in a hospital — a single OR typically costs
\$30–\$100 per minute to run.
If a 90-minute case is scheduled but actually takes 150 minutes, every subsequent case in that
room is delayed, staff overtime costs spike, and patient satisfaction falls.
If a 150-minute slot is booked for a 90-minute case, expensive OR time sits idle.

A well-calibrated duration model reduces both kinds of waste.
Studies consistently show that even reducing scheduling error by 10–15 minutes per case
produces measurable improvements in OR utilisation and staff efficiency.

### What Makes It Hard

Three properties make this problem genuinely difficult:

**1. Right-skewed distribution.** Most surgeries are short (30–120 minutes), but a small
fraction are very long (300–600+ minutes). The target distribution is not symmetric.
Arithmetic mean is pulled upward by the long tail, and errors on long cases dominate
mean-squared metrics.

**2. High within-procedure variability.** Two patients undergoing the same procedure
(say, "laparoscopic cholecystectomy") can differ by 60+ minutes depending on anatomy,
complications, and surgeon experience on that day.

**3. Natural language in a structured world.** The most informative single feature —
the procedure name — is free text ("Left total knee arthroplasty with cement fixation"),
not a clean categorical code.
Converting that text into a number the model can use is the core engineering challenge
of this project.

---

## 1.2 The Dataset

### Overview

```{.graphviz}
digraph Dataset {
    graph [fontsize=20, dpi=150, size="9,5", ratio=auto,
           margin=0.2, nodesep=0.6, ranksep=0.5,
           fontname="DejaVu Sans", bgcolor="transparent"];
    node  [shape=box, style="rounded,filled", fontsize=18,
           fontname="DejaVu Sans", fontcolor=white, margin=0.2];
    edge  [fontsize=16, penwidth=2, arrowsize=1.2,
           color="#F57C00", fontname="DejaVu Sans"];
    rankdir=LR;

    subgraph cluster_raw {
        style=filled; fillcolor="#0D3B6E";
        fontcolor=white; fontname="DejaVu Sans"; fontsize=18;
        label="Raw EHR Extract";
        ehr [label="Surgical case records\n180,370 rows\nSingle institution", fillcolor="#1976D2"];
    }
    subgraph cluster_features {
        style=filled; fillcolor="#1B3A1B";
        fontcolor=white; fontname="DejaVu Sans"; fontsize=18;
        label="Feature Groups";
        struct [label="Structured\n24 pre-op features", fillcolor="#388E3C"];
        text  [label="Free-text\nProcedure name", fillcolor="#388E3C"];
    }
    subgraph cluster_target {
        style=filled; fillcolor="#5C1A00";
        fontcolor=white; fontname="DejaVu Sans"; fontsize=18;
        label="Target";
        dur [label="Surgery duration\n(minutes)", fillcolor="#BF360C"];
    }

    ehr -> struct [label="extract"];
    ehr -> text  [label="extract"];
    ehr -> dur   [label="label"];
}
```

**What to observe**: The raw EHR data splits into two feature types (structured numbers/categories
and free text) plus one target value. This split is not arbitrary — structured features and text
require completely different preprocessing pipelines.

### Structured Features (24 columns)

| Category | Features | Notes |
|---|---|---|
| Patient | Age, sex, ASA physical status, BMI | Pre-operative assessment values |
| Procedure | Surgical specialty, procedure code, laterality | Categorical codes |
| Scheduling | Day of week, time of day, OR room number | Temporal and spatial context |
| Surgeon | Anonymised surgeon ID, case volume quartile | Proxy for experience |
| Hospital | Elective vs. emergency, teaching vs. non-teaching | System-level context |
| History | Number of prior surgeries, prior complications flag | Cumulative patient history |

All 24 features are known **before** the patient enters the OR.
None of them require intraoperative data (anaesthesia time, actual start delay, etc.).
This is the **leakage-free** constraint — using only information that would realistically be
available when the schedule is being built the night before.

### The Free-text Feature

Each record contains one free-text field: the scheduled procedure name as entered by the
booking clerk.
Examples from the dataset:

```
"Right total hip arthroplasty with uncemented fixation"
"Laparoscopic appendectomy"
"Open Whipple procedure (pancreaticoduodenectomy) with portal vein resection"
"Bilateral inguinal hernia repair, mesh"
```

This field is the most predictive single feature in the dataset — a model that reads the
procedure name and nothing else achieves a lower MAE than a model that reads all 24
structured features combined.
The reason is simple: "Whipple procedure" carries vastly more duration information than
"ASA class III, male, age 62."

---

## 1.3 The Target Variable: Surgical Duration

### What Is Measured

The target is the total surgical case time in minutes, measured from the first incision
to wound closure (skin-to-skin time).
It does **not** include anaesthesia induction, patient positioning, or room turnover.

### Distribution Properties

The distribution of surgical durations is strongly right-skewed.
This is a universal property of OR times across institutions.

```{.matplotlib}
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)
# Simulate realistic OR duration distribution (log-normal, matching real dataset stats)
# Mean ~106 min, median ~85 min, right tail to 600+ min
durations = rng.lognormal(mean=4.45, sigma=0.72, size=180_370)
durations = np.clip(durations, 15, 660)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
fig.patch.set_facecolor('#0A0A0A')
for ax in (ax1, ax2, ax3):
    ax.set_facecolor('#0A0A0A')
    for spine in ax.spines.values():
        spine.set_color('#444444')
    ax.tick_params(colors='#CCCCCC', labelsize=11)
    ax.xaxis.label.set_color('#CCCCCC')
    ax.yaxis.label.set_color('#CCCCCC')

# Panel 1: Full histogram
ax1.hist(durations, bins=100, color='#1565C0', edgecolor='none', alpha=0.85)
ax1.axvline(np.mean(durations), color='#F57C00', lw=2, label=f'Mean {np.mean(durations):.0f} min')
ax1.axvline(np.median(durations), color='#2E7D32', lw=2, linestyle='--',
            label=f'Median {np.median(durations):.0f} min')
ax1.set_xlabel('Duration (minutes)')
ax1.set_ylabel('Number of cases')
ax1.set_title('Full distribution', color='#CCCCCC', fontsize=12)
ax1.legend(fontsize=10, facecolor='#1A1A1A', labelcolor='#CCCCCC')

# Panel 2: Log-scale to reveal tail
ax2.hist(durations, bins=100, color='#C62828', edgecolor='none', alpha=0.85)
ax2.set_yscale('log')
ax2.set_xlabel('Duration (minutes)')
ax2.set_ylabel('Count (log scale)')
ax2.set_title('Log-scale y-axis (reveals tail)', color='#CCCCCC', fontsize=12)

# Panel 3: Cumulative distribution
sorted_d = np.sort(durations)
cdf = np.arange(1, len(sorted_d)+1) / len(sorted_d)
ax3.plot(sorted_d, cdf, color='#0097A7', lw=2)
ax3.axhline(0.50, color='#F57C00', lw=1.5, linestyle='--', alpha=0.7, label='50th pct')
ax3.axhline(0.90, color='#C62828', lw=1.5, linestyle='--', alpha=0.7, label='90th pct')
ax3.axhline(0.99, color='#6A1B9A', lw=1.5, linestyle='--', alpha=0.7, label='99th pct')
ax3.set_xlabel('Duration (minutes)')
ax3.set_ylabel('Cumulative fraction')
ax3.set_title('Cumulative distribution', color='#CCCCCC', fontsize=12)
ax3.legend(fontsize=10, facecolor='#1A1A1A', labelcolor='#CCCCCC')

plt.tight_layout()
```

**What to observe in these three panels:**

- **Panel 1** (histogram): The distribution peaks around 85–100 minutes, but the orange mean
  line sits noticeably to the right of the green median. That gap is a signature of right-skew —
  a few very long cases pull the average upward.
- **Panel 2** (log y-axis): Reveals that the extreme tail (300–600 min cases) is not noise.
  There are hundreds of genuine cases that long. Any model must handle these without catastrophic
  errors on the short, common cases.
- **Panel 3** (CDF): 90% of all cases finish within ~260 minutes. The last 10% spans the next
  400 minutes. This is the "heavy tail" we must be careful about when choosing loss functions
  and evaluation metrics.

---

## 1.4 Formalising the Regression Problem

### Intuition (before any symbols)

We have a table of 180,370 rows.
Each row is one surgery.
Each row has 24 numbers/categories (the structured features) plus one text string
(the procedure name).
We want to find a mathematical function that, when given all those inputs, outputs a number
close to the actual surgery duration.
"Close" will be defined precisely using metrics like Mean Absolute Error.
The function is found by adjusting its internal parameters until its predictions match
the training data as well as possible.

### Mathematical Formulation

$$\hat{y}_i = f(\mathbf{x}_i^{\text{struct}},\; \mathbf{e}_i^{\text{text}};\; \boldsymbol{\theta})$$

**Symbol definitions:**

- $i$: index of a single surgical case, $i \in \{1, 2, \ldots, N\}$
- $N = 180{,}370$: total number of cases
- $\hat{y}_i$: the model's predicted duration for case $i$, in minutes
- $y_i$: the true recorded duration for case $i$, in minutes
- $\mathbf{x}_i^{\text{struct}} \in \mathbb{R}^{38}$: the 38-dimensional structured feature vector after one-hot encoding (24 raw features expand to 38 after encoding categoricals)
- $\mathbf{e}_i^{\text{text}} \in \mathbb{R}^{d}$: the text embedding vector for the procedure name; $d$ depends on the encoder used (384 for SentenceBERT, 768 for Bio-ClinicalBERT, 128 for TinySurgicalBERT before projection)
- $\boldsymbol{\theta}$: all learnable parameters of the downstream regression model
- $f(\cdot;\, \boldsymbol{\theta})$: the downstream regression function (Ridge, XGBoost, etc.)

**Term-by-term breakdown:**

1. $\mathbf{x}_i^{\text{struct}}$: The 24 raw structured features become 38 dimensions after
   one-hot encoding of categorical columns (e.g., surgical specialty with 8 unique values
   becomes 8 binary columns). This is a one-time data transformation.

2. $\mathbf{e}_i^{\text{text}}$: The procedure free-text string is converted into a
   fixed-length vector by a pre-trained encoder. This encoding step happens once before
   modelling and is the central contribution of the TinySurgicalBERT work.

3. $f(\cdot;\, \boldsymbol{\theta})$: The regression model takes the combined feature vector
   and outputs a scalar prediction. Training finds $\boldsymbol{\theta}$ that minimises
   prediction error across all training cases.

### The Training Objective

$$\boldsymbol{\theta}^* = \underset{\boldsymbol{\theta}}{\arg\min}\; \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}(y_i,\; \hat{y}_i)$$

- $\boldsymbol{\theta}^*$: the optimal parameter values — what training produces
- $\mathcal{L}(y_i, \hat{y}_i)$: a loss function measuring how wrong prediction $\hat{y}_i$ is (different choices of $\mathcal{L}$ produce Ridge, Lasso, etc. — covered in Section 7)
- $\arg\min$: "the value of $\boldsymbol{\theta}$ that makes the expression as small as possible"

### Numerical Example

Suppose we have 5 cases (toy example):

| Case | True duration $y_i$ | Predicted $\hat{y}_i$ | Error $y_i - \hat{y}_i$ |
|---|---|---|---|
| 1 | 85 min | 91 min | −6 min |
| 2 | 210 min | 195 min | +15 min |
| 3 | 45 min | 52 min | −7 min |
| 4 | 320 min | 298 min | +22 min |
| 5 | 70 min | 68 min | +2 min |

Mean Absolute Error:

$$\text{MAE} = \frac{1}{5}(6 + 15 + 7 + 22 + 2) = \frac{52}{5} = 10.4 \text{ min}$$

**Interpretation**: On average, this toy model is wrong by 10.4 minutes.
Cases scheduled in 15-minute OR blocks would occasionally be misclassified into the wrong block.
In the real project we achieve MAE = 26.38 minutes across 180,370 cases with TinySurgicalBERT + XGBoost.

---

## 1.5 Code: Loading and Inspecting the Dataset

```python
import sqlite3
import pandas as pd
import numpy as np

# Connect to the results database (populated after pipeline runs)
with sqlite3.connect('./results/result.db') as conn:
    # Load the cross-validation metrics for every model/encoding combination
    metrics = pd.read_sql("SELECT * FROM metrics", conn)

# What does the metrics table look like?
print(metrics.head(3).to_string())
# Output:
#    mse   rmse    mae    r2  smape  mean_error  std_error  ci95_low  ci95_high  fold        encoding  n_features    model  train_time_s  infer_time_s
# 0  4171  64.59  46.64  0.65  43.3      -0.27       64.6     -0.94       0.39     0  only_structured           0    ridge        0.0188        0.0011
# 1  2683  51.80  34.95  0.78  28.9      -0.25       51.8     -0.79       0.28     0  only_structured           0  xgboost        2.2707        0.0470

print(f"\nTotal experiments: {len(metrics)}")
# Output: Total experiments: 160

print(f"Encodings: {metrics['encoding'].unique().tolist()}")
# Output: ['only_structured', 'sentencebert', 'clinicalbert', 'tinybert']

print(f"Models: {metrics['model'].unique().tolist()}")
# Output: ['linear', 'ridge', 'lasso', 'elasticnet', 'randomforest', 'xgboost', 'lightgbm', 'mlp']

# Best result
best = metrics.loc[metrics['mae'].idxmin()]
print(f"\nBest single fold: {best['encoding']} + {best['model']}, MAE={best['mae']:.2f}")
# Output: Best single fold: tinybert + xgboost, MAE=26.23
```

---

## Summary

| Concept | Key Takeaway |
|---|---|
| Task type | Supervised regression — output is continuous minutes |
| Dataset size | 180,370 cases — large enough for stable 5-fold CV |
| Feature types | 24 structured (numeric/categorical) + 1 free-text |
| Target distribution | Right-skewed log-normal; median ≈ 85 min, mean ≈ 106 min |
| Core challenge | Converting free-text procedure names into a machine-readable format |
| Best result | MAE = 26.38 min (TinySurgicalBERT + XGBoost) |
