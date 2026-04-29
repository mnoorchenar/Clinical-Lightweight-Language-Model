# TinySurgicalBERT — Section 8: Evaluation Metrics

## Introduction

A regression model produces numbers.
Whether those numbers are "good" depends on how you measure "good."
This section derives all five metrics used in the project — MAE, MSE, RMSE, sMAPE, and R² —
from first principles, including full worked numerical examples and a clear explanation of
when to prefer each one.

---

## 8.1 The Five Metrics at a Glance

```{.graphviz}
digraph Metrics {
    graph [fontsize=20, dpi=150, size="10,5", ratio=auto,
           margin=0.2, nodesep=0.6, ranksep=0.5,
           fontname="DejaVu Sans", bgcolor="transparent"];
    node  [shape=box, style="rounded,filled", fontsize=17,
           fontname="DejaVu Sans", fontcolor=white, margin=0.18];
    edge  [fontsize=14, penwidth=2, arrowsize=1.2,
           color="#F57C00", fontname="DejaVu Sans"];
    rankdir=LR;

    pred [label="Predictions\n{y_hat_1, ..., y_hat_N}", fillcolor="#1976D2", color="#0D3B6E"];
    true [label="True values\n{y_1, ..., y_N}", fillcolor="#1976D2", color="#0D3B6E"];

    subgraph cluster_abs {
        style=filled; fillcolor="#1B3A1B"; fontcolor=white;
        fontname="DejaVu Sans"; fontsize=17; label="Absolute error metrics";
        mae  [label="MAE\n(minutes)", fillcolor="#388E3C"];
        rmse [label="RMSE\n(minutes)", fillcolor="#388E3C"];
    }
    subgraph cluster_sq {
        style=filled; fillcolor="#3E0A6E"; fontcolor=white;
        fontname="DejaVu Sans"; fontsize=17; label="Squared error metric";
        mse [label="MSE\n(min²)", fillcolor="#7B1FA2"];
    }
    subgraph cluster_rel {
        style=filled; fillcolor="#5C1A00"; fontcolor=white;
        fontname="DejaVu Sans"; fontsize=17; label="Relative error metric";
        smape [label="sMAPE\n(%)", fillcolor="#BF360C"];
    }
    subgraph cluster_r2 {
        style=filled; fillcolor="#003333"; fontcolor=white;
        fontname="DejaVu Sans"; fontsize=17; label="Explained variance";
        r2 [label="R²\n(unitless)", fillcolor="#00796B"];
    }

    pred -> mae; true -> mae;
    pred -> mse; true -> mse;
    pred -> rmse; true -> rmse;
    pred -> smape; true -> smape;
    pred -> r2; true -> r2;
}
```

---

## 8.2 Running Example

We will use the same 5 cases throughout all five metric calculations:

| Case $i$ | True $y_i$ | Predicted $\hat{y}_i$ | Error $e_i = y_i - \hat{y}_i$ |
|---|---|---|---|
| 1 | 60 min | 68 min | −8 min |
| 2 | 120 min | 113 min | +7 min |
| 3 | 250 min | 238 min | +12 min |
| 4 | 45 min | 51 min | −6 min |
| 5 | 180 min | 166 min | +14 min |

Mean of true values: $\bar{y} = (60 + 120 + 250 + 45 + 180)/5 = 131$ min

---

## 8.3 Mean Absolute Error (MAE)

### Intuition (before any symbols)

The most intuitive error metric: compute the absolute difference between each prediction
and the truth, then take the average.
"Absolute" means we treat a 10-minute overestimate and a 10-minute underestimate as
equally bad — we only care about magnitude, not direction.
MAE is in the same units as the target (minutes), making it directly interpretable:
"on average, my model is wrong by 26.38 minutes."

### Formula

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

### Symbol definitions

- $N$: number of cases
- $y_i$: the true duration of case $i$
- $\hat{y}_i$: the predicted duration of case $i$
- $|y_i - \hat{y}_i|$: the absolute prediction error — the magnitude of the mistake, always $\geq 0$
- $\sum_{i=1}^{N}(\cdot)$: sum over all $N$ cases
- $\frac{1}{N}(\cdot)$: divide by $N$ to get the average

### Term-by-term breakdown

1. $|y_i - \hat{y}_i|$: The absolute value removes the sign. An error of $-8$ (overestimate) becomes $+8$. This treats both over- and under-predictions symmetrically.
2. $\frac{1}{N}\sum$: Averaging gives one number representing "typical" error.

### Numerical Example

$$\text{MAE} = \frac{|{-8}| + |{+7}| + |{+12}| + |{-6}| + |{+14}|}{5} = \frac{8 + 7 + 12 + 6 + 14}{5} = \frac{47}{5} = 9.4 \text{ min}$$

**Interpretation**: This toy model is wrong by 9.4 minutes on average.
In a real OR scheduling context, 9.4 minutes is within one scheduling block (15 min),
so this would be excellent. Our actual best result: 26.38 minutes.

### When to Use MAE

MAE is the most important metric for OR scheduling because:
- It is in interpretable units (minutes)
- It treats all errors equally (a 5-minute error on a 30-minute case counts the same as a 5-minute error on a 300-minute case)
- It is robust to outliers (a 200-minute error contributes 200 to the sum, not 40,000 like MSE)

---

## 8.4 Mean Squared Error (MSE)

### Intuition

MSE squares each error before averaging.
This has two effects: it removes the sign (like MAE) **and** it disproportionately penalises
large errors.
A 10-minute error contributes 100 to MSE; a 20-minute error contributes 400 — four times as much
for twice the error.
MSE is useful when large errors are especially unacceptable (a 3-hour scheduling overrun is
more than three times as bad as a 1-hour overrun).

### Formula

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

### Symbol definitions

All symbols as in MAE, except:
- $(y_i - \hat{y}_i)^2$: the **squared** error — always non-negative, amplifies large mistakes
- Units: minutes² (not directly interpretable as time)

### Term-by-term breakdown

1. $(y_i - \hat{y}_i)^2$: Squaring achieves sign removal AND outlier amplification.
   An error of $-8$ becomes $+64$; an error of $+14$ becomes $+196$.
   The larger error (14) contributes 3× more than the smaller error (8), even though
   it is only 1.75× larger — this is the "outlier penalty" of MSE.
2. $\frac{1}{N}\sum$: Average over all cases.

### Numerical Example

$$\text{MSE} = \frac{(-8)^2 + (+7)^2 + (+12)^2 + (-6)^2 + (+14)^2}{5}$$
$$= \frac{64 + 49 + 144 + 36 + 196}{5} = \frac{489}{5} = 97.8 \text{ min}^2$$

**Interpretation**: MSE = 97.8 min² is not directly interpretable (what is a "squared minute"?).
That is why RMSE is reported instead.

---

## 8.5 Root Mean Squared Error (RMSE)

### Intuition

RMSE takes the square root of MSE, returning the metric to the original unit (minutes).
It retains MSE's property of penalising large errors more than small ones, but is now
interpretable as "a typical error in minutes."

### Formula

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2} = \sqrt{\text{MSE}}$$

### Numerical Example

$$\text{RMSE} = \sqrt{97.8} = 9.89 \text{ min}$$

**Comparison to MAE**: RMSE (9.89) > MAE (9.4).
RMSE is always $\geq$ MAE, and the gap grows as the distribution of errors becomes more skewed.
Here the two largest errors (12, 14) pull RMSE above MAE.

### When RMSE > MAE is Informative

If RMSE is much larger than MAE, it signals that the model makes occasional very large errors
even if most predictions are accurate.
For OR scheduling, this matters: a model with MAE=26 min but occasional 300-minute errors
is far more disruptive than a model with MAE=28 min and a maximum error of 60 minutes.

```{.matplotlib}
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)
# Two hypothetical error distributions with similar MAE but different RMSE
errors_A = rng.normal(0, 20, 1000)                          # MAE ≈ 16, RMSE ≈ 20 (symmetric)
errors_B = np.concatenate([rng.normal(0, 15, 950),          # mostly small errors
                            rng.normal(0, 80, 50)])          # 5% large errors  → RMSE >> MAE

mae_A = np.mean(np.abs(errors_A)); rmse_A = np.sqrt(np.mean(errors_A**2))
mae_B = np.mean(np.abs(errors_B)); rmse_B = np.sqrt(np.mean(errors_B**2))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor('#0A0A0A')
for ax in (ax1, ax2):
    ax.set_facecolor('#0A0A0A')
    for spine in ax.spines.values(): spine.set_color('#444444')
    ax.tick_params(colors='#CCCCCC', labelsize=11)
    ax.xaxis.label.set_color('#CCCCCC')
    ax.yaxis.label.set_color('#CCCCCC')

bins = np.linspace(-300, 300, 80)
ax1.hist(errors_A, bins=bins, color='#1565C0', edgecolor='none', alpha=0.8)
ax1.axvline(mae_A,  color='#2E7D32', lw=2, label=f'MAE = {mae_A:.1f}')
ax1.axvline(rmse_A, color='#F57C00', lw=2, linestyle='--', label=f'RMSE = {rmse_A:.1f}')
ax1.set_title('Model A: well-behaved errors', color='#CCCCCC', fontsize=11)
ax1.set_xlabel('Prediction error (min)')
ax1.legend(fontsize=11, facecolor='#1A1A1A', labelcolor='#CCCCCC')

ax2.hist(errors_B, bins=bins, color='#C62828', edgecolor='none', alpha=0.8)
ax2.axvline(mae_B,  color='#2E7D32', lw=2, label=f'MAE = {mae_B:.1f}')
ax2.axvline(rmse_B, color='#F57C00', lw=2, linestyle='--', label=f'RMSE = {rmse_B:.1f}')
ax2.set_title('Model B: 5% catastrophic errors (RMSE >> MAE)', color='#CCCCCC', fontsize=11)
ax2.set_xlabel('Prediction error (min)')
ax2.legend(fontsize=11, facecolor='#1A1A1A', labelcolor='#CCCCCC')

plt.tight_layout()
```

**What to observe**: Both models have similar MAE (left ≈ right).
But Model B's RMSE is roughly 2× its MAE, revealing the hidden tail of catastrophic errors.
A scheduler comparing only MAE would choose Model B; adding RMSE reveals its dangerous tail.

---

## 8.6 Symmetric Mean Absolute Percentage Error (sMAPE)

### Intuition

Sometimes we care about **relative** error, not absolute.
Being wrong by 10 minutes on a 20-minute procedure is catastrophic (50% error);
being wrong by 10 minutes on a 300-minute procedure is minor (~3% error).
sMAPE scales the error by the average of the true and predicted values,
making it a percentage.

The "symmetric" version uses the average of $y$ and $\hat{y}$ in the denominator,
which avoids dividing by zero and treats over- and under-predictions symmetrically.

### Formula

$$\text{sMAPE} = \frac{100\%}{N} \sum_{i=1}^{N} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}$$

### Symbol definitions

- $\frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}$: the absolute error normalised by the average of true and predicted values
- $100\%$: converts the ratio to a percentage
- Note: sMAPE is bounded in $[0\%, 200\%]$ — a value of 200% means one of the two values is zero

### Term-by-term breakdown

1. Numerator $|y_i - \hat{y}_i|$: the absolute error (same as in MAE).
2. Denominator $(|y_i| + |\hat{y}_i|)/2$: the average magnitude of true and predicted.
   Using the average (rather than just $y_i$) prevents the metric from blowing up when
   $y_i$ is very small.
3. $\frac{100\%}{N}\sum$: converts to percentage and averages.

### Numerical Example

| Case | $y_i$ | $\hat{y}_i$ | $|e_i|$ | $(y_i + \hat{y}_i)/2$ | Ratio |
|---|---|---|---|---|---|
| 1 | 60 | 68 | 8 | (60+68)/2 = 64.0 | 8/64.0 = 0.125 |
| 2 | 120 | 113 | 7 | (120+113)/2 = 116.5 | 7/116.5 = 0.0601 |
| 3 | 250 | 238 | 12 | (250+238)/2 = 244.0 | 12/244.0 = 0.0492 |
| 4 | 45 | 51 | 6 | (45+51)/2 = 48.0 | 6/48.0 = 0.1250 |
| 5 | 180 | 166 | 14 | (180+166)/2 = 173.0 | 14/173.0 = 0.0809 |

$$\text{sMAPE} = \frac{100\%}{5}(0.125 + 0.0601 + 0.0492 + 0.1250 + 0.0809)$$
$$= 20\% \times 0.4402 = 8.80\%$$

**Interpretation**: The toy model is wrong by 8.8% of the procedure duration on average.
Our best real model achieves sMAPE ≈ 21.6% — meaning on a 100-minute procedure,
the prediction is off by about 21.6 minutes on average.

---

## 8.7 Coefficient of Determination (R²)

### Intuition

R² measures what fraction of the total variation in surgical duration is explained by the model.
An R² of 1.0 means perfect prediction.
An R² of 0.0 means the model is no better than always predicting the mean duration.
An R² of 0.854 (our best result) means the model explains 85.4% of the variation —
only 14.6% of the variance in surgical duration remains unexplained.

**Real-world analogy**: imagine explaining why a student's exam score varies.
If you know nothing (predict the class average every time), you explain 0% of the variance.
If you know their study hours, you might explain 60%.
If you also know their prior grades, you might explain 80%.
R² measures how much your model improves on the "always predict the mean" baseline.

### Formula

$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$$

where:

$$\text{SS}_{\text{res}} = \sum_{i=1}^{N}(y_i - \hat{y}_i)^2, \qquad \text{SS}_{\text{tot}} = \sum_{i=1}^{N}(y_i - \bar{y})^2$$

### Symbol definitions

- $\text{SS}_{\text{res}}$: **residual sum of squares** — total squared error of the model's predictions
- $\text{SS}_{\text{tot}}$: **total sum of squares** — total squared variation of the target around its mean
- $\bar{y} = \frac{1}{N}\sum_{i=1}^{N} y_i$: the mean of the true durations
- $1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$: the fraction of variance explained

### Term-by-term breakdown

1. **$\text{SS}_{\text{tot}}$**: This is the MSE of a "null model" that always predicts $\bar{y}$.
   It represents the maximum possible error a sensible model would make.
2. **$\text{SS}_{\text{res}}$**: This is the actual squared error of our model.
   If our model is perfect ($\hat{y}_i = y_i$), $\text{SS}_{\text{res}} = 0$ and $R^2 = 1$.
3. **$\frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$**: The fraction of variance NOT explained.
   Subtracting from 1 gives the fraction that IS explained.
4. **R² can be negative**: if $\text{SS}_{\text{res}} > \text{SS}_{\text{tot}}$, the model is worse
   than the null model — it makes predictions so bad they add variance rather than reducing it.

### Numerical Example

$\bar{y} = 131$ min (computed earlier).

**Step 1 — Compute SS_tot:**

$$\text{SS}_{\text{tot}} = (60-131)^2 + (120-131)^2 + (250-131)^2 + (45-131)^2 + (180-131)^2$$
$$= (-71)^2 + (-11)^2 + (119)^2 + (-86)^2 + (49)^2$$
$$= 5041 + 121 + 14161 + 7396 + 2401 = 29120$$

**Step 2 — Compute SS_res:**

$$\text{SS}_{\text{res}} = (-8)^2 + 7^2 + 12^2 + (-6)^2 + 14^2 = 64 + 49 + 144 + 36 + 196 = 489$$

**Step 3 — Compute R²:**

$$R^2 = 1 - \frac{489}{29120} = 1 - 0.01680 = 0.9832$$

**Interpretation**: This toy model explains 98.3% of the variance in surgical duration.
That is unrealistically high for a 5-case example; our real model achieves R² = 0.854 (85.4%)
on 36,074 validation cases — a genuinely strong result for a complex clinical regression task.

---

## 8.8 All Metrics Together: A Visual Summary

```{.matplotlib}
import matplotlib.pyplot as plt
import numpy as np

# All four text encodings, XGBoost only, from result.db
encodings = ['Structured\nOnly', 'SentenceBERT', 'Bio-Clinical\nBERT', 'TinySurgical\nBERT']
mae   = [34.80, 26.35, 26.40, 26.38]
rmse  = [51.77, 41.89, 41.93, 41.87]
smape = [28.84, 21.56, 21.62, 21.61]
r2    = [0.777, 0.854, 0.854, 0.854]

x = np.arange(len(encodings))
colors = ['#6A1B9A', '#1565C0', '#F57C00', '#2E7D32']

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
fig.patch.set_facecolor('#0A0A0A')

for ax, values, ylabel, title in zip(
    axes,
    [mae, rmse, smape, r2],
    ['MAE (min)', 'RMSE (min)', 'sMAPE (%)', 'R²'],
    ['MAE (lower better)', 'RMSE (lower better)', 'sMAPE (lower better)', 'R² (higher better)']
):
    ax.set_facecolor('#0A0A0A')
    for spine in ax.spines.values(): spine.set_color('#444444')
    ax.tick_params(colors='#CCCCCC', labelsize=9)
    bars = ax.bar(x, values, color=colors, edgecolor='none', alpha=0.88, width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(encodings, color='#CCCCCC', fontsize=8.5)
    ax.set_ylabel(ylabel, color='#CCCCCC', fontsize=10)
    ax.set_title(title, color='#CCCCCC', fontsize=10)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                f'{v:.2f}', ha='center', color='#CCCCCC', fontsize=9)

plt.tight_layout()
```

**What to observe**: Across all four metrics, the three BERT-based encodings (right three bars)
are dramatically better than Structured Only (first bar).
Among the three BERT encodings, differences are tiny — TinySurgicalBERT matches Bio-ClinicalBERT
almost exactly on every metric, validating that 614× compression does not sacrifice accuracy.

---

## Summary

| Metric | Formula | Units | Best value | Project result |
|---|---|---|---|---|
| MAE | $\frac{1}{N}\sum|y - \hat{y}|$ | minutes | 0 | 26.38 min |
| MSE | $\frac{1}{N}\sum(y - \hat{y})^2$ | min² | 0 | 1,754 min² |
| RMSE | $\sqrt{\text{MSE}}$ | minutes | 0 | 41.87 min |
| sMAPE | $\frac{100\%}{N}\sum\frac{|y-\hat{y}|}{(|y|+|\hat{y}|)/2}$ | % | 0% | 21.61% |
| R² | $1 - \text{SS}_\text{res}/\text{SS}_\text{tot}$ | unitless | 1.0 | 0.854 |
