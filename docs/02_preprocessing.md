# TinySurgicalBERT — Section 2: Data Preprocessing

## Introduction

Raw hospital data is never ready for a machine learning model.
It contains errors, leaking future information, inconsistent categories, missing values,
and time fields encoded as plain integers (e.g., hour = 14) that obscure cyclical patterns.
This section walks through every preprocessing step in Stage 1 of the pipeline:
why it is needed, exactly how it works mathematically, and a worked numerical example for each.

---

## 2.1 The Preprocessing Pipeline at a Glance

```{.graphviz}
digraph Preprocessing {
    graph [fontsize=20, dpi=150, size="8,10", ratio=auto,
           margin=0.2, nodesep=0.5, ranksep=0.45,
           fontname="DejaVu Sans", bgcolor="transparent"];
    node  [shape=box, style="rounded,filled", fontsize=18,
           fontname="DejaVu Sans", fontcolor=white, margin=0.18];
    edge  [fontsize=16, penwidth=2, arrowsize=1.2,
           color="#F57C00", fontname="DejaVu Sans"];
    rankdir=TB;

    raw [label="Raw EHR Data\n180,370 rows", fillcolor="#1976D2", color="#0D3B6E"];

    subgraph cluster_s1a {
        style=filled; fillcolor="#1B3A1B"; fontcolor=white;
        fontname="DejaVu Sans"; fontsize=18; label="Step 1 — Leakage Removal";
        leak [label="Remove intraoperative columns\n(actual start, anaesthesia time,\nroom turnover)", fillcolor="#388E3C"];
    }
    subgraph cluster_s1b {
        style=filled; fillcolor="#3E0A6E"; fontcolor=white;
        fontname="DejaVu Sans"; fontsize=18; label="Step 2 — Implausibility Filter";
        filt [label="Drop: duration < 15 min\nor duration > 660 min", fillcolor="#7B1FA2"];
    }
    subgraph cluster_s1c {
        style=filled; fillcolor="#5C1A00"; fontcolor=white;
        fontname="DejaVu Sans"; fontsize=18; label="Step 3 — Temporal Encoding";
        time [label="Hour-of-day and day-of-week\nsin/cos cyclic encoding", fillcolor="#BF360C"];
    }
    subgraph cluster_s1d {
        style=filled; fillcolor="#003333"; fontcolor=white;
        fontname="DejaVu Sans"; fontsize=18; label="Step 4 — Categorical Harmonisation";
        cat [label="Surgeon ID, room, specialty\nunified label encoding", fillcolor="#00796B"];
    }
    subgraph cluster_s1e {
        style=filled; fillcolor="#0D3B6E"; fontcolor=white;
        fontname="DejaVu Sans"; fontsize=18; label="Step 5 — Missing Value Strategy";
        miss [label="Median imputation (fold-wise)\nto prevent leakage", fillcolor="#1976D2"];
    }

    clean [label="Clean Cohort\n180,370 cases\n24 pre-op features", fillcolor="#1976D2", color="#0D3B6E"];

    raw   -> leak -> filt -> time -> cat -> miss -> clean;
}
```

**What to observe**: Each step removes one specific problem from the data.
The order matters — leakage removal comes first because any feature that inadvertently
contains future information would corrupt all subsequent steps.

---

## 2.2 Leakage Prevention

### What It Is

Data leakage occurs when information that would not be available at prediction time
sneaks into the feature set during training.
A leaky model learns to exploit this "cheat" information, achieving impressive training scores
that completely collapse on real deployment.

**Real-world analogy**: imagine you are training a model to predict whether a student will
pass an exam, and you accidentally include "number of correct answers on the exam" as a feature.
The model learns that feature perfectly, achieves 100% accuracy, and is completely useless
when deployed — because you do not know the exam answers before grading it.

In an OR context, the leaky features are any times or events recorded *during* the surgery:
anaesthesia start time, actual room entry time, intraoperative complication flags, actual end time.
All of these are recorded *after* the prediction must be made, so they must be excluded.

### Leaky vs. Safe Features

| Feature | Leaky? | Why |
|---|---|---|
| Scheduled start time | ✅ Safe | Known at booking time |
| Day of week | ✅ Safe | Known at booking time |
| Actual anaesthesia start | ❌ Leaky | Recorded in the OR — not available pre-op |
| ASA physical status | ✅ Safe | Set at pre-operative assessment |
| Room turnover time | ❌ Leaky | Measured between actual cases |
| Surgeon ID | ✅ Safe | Assigned at booking time |
| Intraoperative complication flag | ❌ Leaky | Set during surgery |

The pipeline drops all leaky columns before any further processing.

---

## 2.3 Implausibility Filtering

### What It Is

Some records in the raw EHR are data entry errors: cases coded as 3 minutes long
(a scheduling entry that was immediately cancelled) or 700 minutes long (a multi-day complex
procedure that wrapped around midnight and was recorded twice).
These outliers are not real surgeries we want to predict; they are artefacts of the data
collection system.

We drop records where the recorded duration is less than 15 minutes or greater than 660 minutes
(11 hours).
The 15-minute floor reflects the minimum plausible skin-to-skin time for any real procedure.
The 660-minute ceiling captures the 99.9th percentile of genuinely long cases while
excluding obvious errors.

### Mathematical Definition

A case $i$ is retained if and only if:

$$15 \leq y_i \leq 660$$

where $y_i$ is the recorded duration in minutes.
Cases outside this interval are removed entirely (not imputed).

**Why not impute instead of drop?** Imputing a duration of 3 minutes to the median makes
no sense — a 3-minute record is a *data entry error*, not a patient with an unusually short
surgery. The label itself is wrong, so no downstream imputation can fix it.

---

## 2.4 Cyclic Temporal Encoding

### What It Is and Why It Exists

The raw dataset contains hour-of-day (0–23) and day-of-week (0–6) as plain integers.
Feeding those raw integers to a model creates a serious problem:
the model would see Monday (0) and Friday (4) as further apart than Monday (0) and Saturday (5),
even though Saturday is adjacent to Sunday (6) which is adjacent to Monday (0).
In other words, time is **circular** — after midnight comes 1 AM, not a number far from 23.
A plain integer does not capture this.

**Real-world analogy**: imagine measuring your position on a clock face as "the number of the
hour hand position" (1 through 12). At 11 o'clock you are "close" to midnight, but the
integer 11 is far from the integer 1 (which represents 1 AM). A straight-line distance on
integers does not respect the circular geometry of time.

### Mathematical Foundation

**Intuition (no symbols first)**: We convert each time value into two numbers — a sine and a
cosine — that together describe a point on a unit circle.
As the time advances, the point moves smoothly around the circle, and importantly,
23:00 and 00:00 are adjacent points on that circle just like 11:00 and 12:00.
No abrupt jump from "large integer" back to "small integer."

**Formula — cyclic encoding for a periodic variable $t$ with period $P$:**

$$\sin\_t = \sin\!\left(\frac{2\pi \cdot t}{P}\right), \qquad \cos\_t = \cos\!\left(\frac{2\pi \cdot t}{P}\right)$$

**Symbol definitions:**

- $t$: the raw integer value of the time feature (e.g., hour $\in \{0, 1, \ldots, 23\}$, or day $\in \{0, 1, \ldots, 6\}$)
- $P$: the period of the cycle — 24 for hours-of-day, 7 for day-of-week
- $2\pi$: one full revolution of a circle (in radians)
- $\frac{2\pi \cdot t}{P}$: the angle in radians corresponding to position $t$ within the full cycle
- $\sin(\cdot)$, $\cos(\cdot)$: trigonometric functions that map any angle to a value in $[-1, +1]$
- $\sin\_t$, $\cos\_t$: the two new features that replace the original integer $t$

**Term-by-term breakdown:**

1. $\frac{2\pi}{P}$: This is the "step size" in radians per unit of $t$.
   For hours, each additional hour advances the angle by $\frac{2\pi}{24} = 0.2618$ radians.
   After 24 steps (one full day), the angle returns exactly to 0, which is what we want.

2. $\sin(\cdot)$ and $\cos(\cdot)$ together: You need **both** sine and cosine because
   sine alone is ambiguous — $\sin(90°) = \sin(90°)$ is fine, but $\sin(30°) = \sin(150°)$,
   meaning two different hours would produce the same sine value.
   Together, the pair $(\sin, \cos)$ uniquely identifies every point on the circle.

### Numerical Example

Encoding three different hours: midnight (0:00), noon (12:00), and 11 PM (23:00).

**Period** $P = 24$.

**Hour 0 (midnight):**

$$\sin\_0 = \sin\!\left(\frac{2\pi \cdot 0}{24}\right) = \sin(0) = 0.000$$

$$\cos\_0 = \cos\!\left(\frac{2\pi \cdot 0}{24}\right) = \cos(0) = 1.000$$

**Hour 12 (noon):**

$$\sin\_{12} = \sin\!\left(\frac{2\pi \cdot 12}{24}\right) = \sin(\pi) = 0.000$$

$$\cos\_{12} = \cos\!\left(\frac{2\pi \cdot 12}{24}\right) = \cos(\pi) = -1.000$$

**Hour 23 (11 PM):**

$$\sin\_{23} = \sin\!\left(\frac{2\pi \cdot 23}{24}\right) = \sin(5.890) = -0.259$$

$$\cos\_{23} = \cos\!\left(\frac{2\pi \cdot 23}{24}\right) = \cos(5.890) = +0.966$$

| Hour | Raw integer | sin | cos | Euclidean distance to midnight |
|---|---|---|---|---|
| 0 (midnight) | 0 | 0.000 | 1.000 | — |
| 12 (noon) | 12 | 0.000 | −1.000 | 2.000 |
| 23 (11 PM) | 23 | −0.259 | 0.966 | **0.261** |

**Interpretation**: Hour 23 is Euclidean distance 0.261 from midnight in the encoded space,
correctly reflecting that 11 PM is close to midnight.
Without cyclic encoding, the raw integers would place 23 as distance 23 from 0 — the worst
possible distance, when actually they are adjacent in time.

### Visualisation of Cyclic Encoding

```{.matplotlib}
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor('#0A0A0A')
for ax in (ax1, ax2):
    ax.set_facecolor('#0A0A0A')
    for spine in ax.spines.values():
        spine.set_color('#444444')
    ax.tick_params(colors='#CCCCCC', labelsize=11)
    ax.xaxis.label.set_color('#CCCCCC')
    ax.yaxis.label.set_color('#CCCCCC')

hours = np.arange(24)
P = 24
sin_h = np.sin(2 * np.pi * hours / P)
cos_h = np.cos(2 * np.pi * hours / P)

# Left: sin and cos as functions of hour
ax1.plot(hours, sin_h, color='#1565C0', lw=2.5, marker='o', markersize=6, label='sin(hour)')
ax1.plot(hours, cos_h, color='#F57C00', lw=2.5, marker='s', markersize=6, label='cos(hour)')
ax1.axhline(0, color='#444444', lw=1)
ax1.set_xlabel('Hour of day (raw integer)')
ax1.set_ylabel('Encoded value')
ax1.set_title('Cyclic encoding: 24-hour clock', color='#CCCCCC', fontsize=12)
ax1.set_xticks([0, 6, 12, 18, 23])
ax1.legend(fontsize=11, facecolor='#1A1A1A', labelcolor='#CCCCCC')

# Right: unit circle — hours as points on a circle
theta = 2 * np.pi * hours / P
ax2.scatter(cos_h, sin_h, c=hours, cmap='plasma', s=80, zorder=3)
circle = plt.Circle((0, 0), 1, fill=False, color='#444444', lw=1.5)
ax2.add_patch(circle)
for h in [0, 6, 12, 18]:
    th = 2 * np.pi * h / P
    ax2.annotate(f'{h}h', (np.cos(th)*1.15, np.sin(th)*1.15),
                 color='#CCCCCC', fontsize=10, ha='center')
ax2.scatter([cos_h[0]], [sin_h[0]], color='#2E7D32', s=180, zorder=5, label='Midnight (0h)')
ax2.scatter([cos_h[23]], [sin_h[23]], color='#C62828', s=180, zorder=5, label='11 PM (23h)')
ax2.set_xlim(-1.4, 1.4); ax2.set_ylim(-1.4, 1.4)
ax2.set_aspect('equal')
ax2.set_title('Hours as points on unit circle', color='#CCCCCC', fontsize=12)
ax2.legend(fontsize=11, facecolor='#1A1A1A', labelcolor='#CCCCCC')

plt.tight_layout()
```

**What to observe:**

- **Left panel**: The sine and cosine wave smoothly across the 24 hours.
  Notice that hour 23 and hour 0 are adjacent values on both waves — no abrupt jump.
- **Right panel**: Each hour is a dot on a circle. Midnight (green) and 11 PM (red) are
  next to each other on the circle, correctly capturing their temporal proximity.
  A raw integer would place them at opposite ends of a number line.

---

## 2.5 Categorical Harmonisation

### What It Is

Surgeon IDs (e.g., "SRG-0047"), OR room numbers (e.g., "OR-12"), and surgical specialties
(e.g., "Orthopaedics") arrive as raw strings.
Machine learning models cannot process strings directly — they need numbers.
Categorical harmonisation converts every unique string value into a consistent integer label,
then feeds those integer labels into one-hot encoding.

**One-hot encoding**: a categorical variable with $K$ unique values becomes $K$ binary columns.
For example, specialty with 8 values becomes 8 columns, with exactly one `1` per row.

$$\text{specialty} = \text{"Orthopaedics"} \;\longrightarrow\; [0, 0, 0, 1, 0, 0, 0, 0]$$

**Why not just use the raw integer labels?**
If you encode "Orthopaedics"=3 and "General Surgery"=4, the model treats them as numerically
close. But specialties have no natural ordering — there is no sense in which Orthopaedics is
"one unit more" than General Surgery.
One-hot encoding removes this false ordering.

### Mathematical Formulation

For a categorical feature $c$ with $K$ unique values $\{v_1, v_2, \ldots, v_K\}$,
the one-hot encoding is the indicator vector:

$$\mathbf{z}(c) = [\mathbb{1}(c = v_1),\; \mathbb{1}(c = v_2),\; \ldots,\; \mathbb{1}(c = v_K)] \in \{0,1\}^K$$

**Symbol definitions:**

- $c$: the observed category value for a particular case
- $v_k$: the $k$-th unique category in the vocabulary (in fixed order)
- $\mathbb{1}(c = v_k)$: the indicator function — equals 1 if $c$ equals $v_k$, else 0
- $\mathbf{z}(c)$: the resulting $K$-dimensional binary vector

**Constraint**: $\sum_{k=1}^{K} \mathbb{1}(c = v_k) = 1$ — exactly one element is 1.

### Numerical Example

Surgical specialty with 4 unique values: General, Ortho, Cardiac, Neuro.

| Specialty | Raw label | One-hot $\mathbf{z}$ |
|---|---|---|
| General | 0 | [1, 0, 0, 0] |
| Ortho | 1 | [0, 1, 0, 0] |
| Cardiac | 2 | [0, 0, 1, 0] |
| Neuro | 3 | [0, 0, 0, 1] |

**Interpretation**: A case in Cardiac surgery has vector $[0, 0, 1, 0]$.
The model sees two non-zero columns for two cases in different specialties — and treats each specialty
as its own independent binary feature rather than placing them on a false numeric scale.

---

## 2.6 Missing Value Imputation (Fold-Wise)

### What It Is

Not every patient record is complete.
Some fields — BMI, prior surgery count, ASA status — are occasionally missing.
We fill missing values with the **median** of the training fold (not the full dataset).

**Why fold-wise?** If you compute the median across all 180,370 cases and then split into folds,
you have used information from the validation fold to fill in training values.
That is a subtle form of data leakage.
The correct procedure is: for each fold, compute the median using only the training cases in that fold,
then apply that same median to both training and validation sets.

### Mathematical Formulation

For fold $k$ and feature $j$, the imputation value is:

$$\mu_j^{(k)} = \text{median}\!\left(\{x_{ij} : i \in \mathcal{D}_{\text{train}}^{(k)},\; x_{ij} \neq \text{NaN}\}\right)$$

**Symbol definitions:**

- $k$: fold index, $k \in \{0, 1, 2, 3, 4\}$
- $j$: feature index (column)
- $x_{ij}$: the value of feature $j$ for case $i$
- $\mathcal{D}_{\text{train}}^{(k)}$: the set of case indices in the training partition of fold $k$
- $\mu_j^{(k)}$: the fold-specific imputation value for feature $j$

Then every missing value in fold $k$ (training or validation) is replaced by $\mu_j^{(k)}$.

**Term-by-term breakdown:**

1. The inner set $\{x_{ij} : i \in \mathcal{D}_{\text{train}}^{(k)}, x_{ij} \neq \text{NaN}\}$:
   collect all non-missing values of feature $j$ in the training fold only.
2. $\text{median}(\cdot)$: the middle value when sorted. More robust than mean for skewed features
   like BMI and surgical case count.
3. Applying $\mu_j^{(k)}$ to validation: the validation imputation value is derived from training
   data only, so no future information enters the validation assessment.

### Numerical Example

Feature: "number of prior surgeries" — 5 training cases, 2 missing.

Training values (excluding NaN): $\{0, 1, 3, 5\}$ (sorted)

$$\mu = \text{median}(0, 1, 3, 5) = \frac{1 + 3}{2} = 2.0$$

The two missing training values and any missing values in the validation fold are filled with 2.

**Interpretation**: The patient's prior surgery count is treated as 2 — the typical value among
cases with known history — rather than 0 (which would incorrectly imply "never had surgery").

---

## 2.7 Code: The Preprocessing Steps

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# --- Step 1: Load raw data and drop leaky columns ---
df = pd.read_csv('./data/surgical_cases.csv')

LEAKY_COLS = ['actual_start_time', 'anaesthesia_start', 'room_turnover_mins',
              'intraop_complication']
df = df.drop(columns=LEAKY_COLS, errors='ignore')
# Output: 180,370 rows × (original columns - 4 leaky columns)

# --- Step 2: Implausibility filter ---
df = df[(df['duration_min'] >= 15) & (df['duration_min'] <= 660)]
# Output: typically removes < 0.5% of rows

# --- Step 3: Cyclic temporal encoding ---
def cyclic_encode(series, period):
    """Convert a periodic integer feature into sin/cos pair."""
    radians = 2 * np.pi * series / period
    return np.sin(radians), np.cos(radians)

sin_hour, cos_hour = cyclic_encode(df['scheduled_hour'], period=24)
sin_dow,  cos_dow  = cyclic_encode(df['day_of_week'],    period=7)

df['sin_hour'] = sin_hour   # replaces raw 'scheduled_hour'
df['cos_hour'] = cos_hour
df['sin_dow']  = sin_dow    # replaces raw 'day_of_week'
df['cos_dow']  = cos_dow
df = df.drop(columns=['scheduled_hour', 'day_of_week'])

# --- Step 4: One-hot encode categorical columns ---
CAT_COLS = ['surgical_specialty', 'or_room', 'laterality', 'elective_flag']
enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# fit on training data only (shown here for clarity; actual code is fold-wise)
ohe_matrix = enc.fit_transform(df[CAT_COLS])
ohe_df = pd.DataFrame(ohe_matrix, columns=enc.get_feature_names_out(CAT_COLS))
df = pd.concat([df.drop(columns=CAT_COLS), ohe_df], axis=1)
# Output: each categorical column is replaced by K binary columns

# --- Step 5: Fold-wise imputation (shown for one fold) ---
# In the pipeline this runs inside the 5-fold loop
train_idx = df.index[:144_296]  # example split
val_idx   = df.index[144_296:]

medians = df.loc[train_idx].median(numeric_only=True)  # computed on training only
df_filled = df.copy()
df_filled = df_filled.fillna(medians)  # applied to both train and val
# Output: zero NaN values in df_filled

print(df_filled.shape)
# Output: (180370, 38)  -- 24 original + cyclic expansions + OHE expansions - removed cols
```

---

## Summary

| Step | Input problem | Solution | Mathematical tool |
|---|---|---|---|
| Leakage removal | Future information in features | Drop intraoperative columns | Set membership filter |
| Implausibility filter | Data entry errors in target | Drop $y < 15$ or $y > 660$ | Interval constraint |
| Cyclic encoding | Time integers create false distances | $\sin/\cos$ projection | Trigonometric mapping |
| One-hot encoding | Categorical strings as false numbers | Binary indicator vectors | $\mathbb{1}(\cdot)$ function |
| Fold-wise imputation | Missing values leaking from val to train | Median from training fold only | Fold-aware aggregation |
