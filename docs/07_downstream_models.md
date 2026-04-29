# TinySurgicalBERT — Section 7: The 8 Downstream Regression Models

## Introduction

Once features are assembled, any regression model can be plugged in.
We evaluate eight models spanning three families: linear, tree-ensemble, and neural network.
This section gives each model the full five-component treatment: intuition, mathematics,
numerical example, and code.
Reading all eight in order builds a mental map of how model complexity relates to predictive power.

---

## 7.1 The Model Family Map

```{.graphviz}
digraph Models {
    graph [fontsize=20, dpi=150, size="10,7", ratio=auto,
           margin=0.2, nodesep=0.5, ranksep=0.5,
           fontname="DejaVu Sans", bgcolor="transparent"];
    node  [shape=box, style="rounded,filled", fontsize=17,
           fontname="DejaVu Sans", fontcolor=white, margin=0.18];
    edge  [fontsize=15, penwidth=2, arrowsize=1.2,
           color="#F57C00", fontname="DejaVu Sans"];
    rankdir=LR;

    feat [label="Input\n294-d feature vector", fillcolor="#1976D2", color="#0D3B6E"];

    subgraph cluster_linear {
        style=filled; fillcolor="#1B3A1B"; fontcolor=white;
        fontname="DejaVu Sans"; fontsize=17; label="Linear Family";
        lr  [label="Linear Reg.", fillcolor="#388E3C"];
        rid [label="Ridge", fillcolor="#388E3C"];
        las [label="Lasso", fillcolor="#388E3C"];
        en  [label="ElasticNet", fillcolor="#388E3C"];
    }
    subgraph cluster_tree {
        style=filled; fillcolor="#3E0A6E"; fontcolor=white;
        fontname="DejaVu Sans"; fontsize=17; label="Tree Ensemble Family";
        rf  [label="Random Forest", fillcolor="#7B1FA2"];
        xgb [label="XGBoost", fillcolor="#7B1FA2"];
        lgb [label="LightGBM", fillcolor="#7B1FA2"];
    }
    subgraph cluster_nn {
        style=filled; fillcolor="#5C1A00"; fontcolor=white;
        fontname="DejaVu Sans"; fontsize=17; label="Neural Network";
        mlp [label="MLP", fillcolor="#BF360C"];
    }
    pred [label="Predicted duration\n(minutes)", fillcolor="#00796B", color="#003333"];

    feat -> lr; feat -> rid; feat -> las; feat -> en;
    feat -> rf; feat -> xgb; feat -> lgb;
    feat -> mlp;
    lr -> pred; rid -> pred; las -> pred; en -> pred;
    rf -> pred; xgb -> pred; lgb -> pred;
    mlp -> pred;
}
```

---

## 7.2 Linear Regression

### Intuition

Linear regression assumes the output is a weighted sum of the input features.
It is the simplest possible model: "each feature contributes independently and proportionally
to the output." This assumption is often wrong in complex settings, but it provides a useful baseline.

**Objective function:**

$$\mathcal{L}_{\text{OLS}}(\boldsymbol{w}) = \sum_{i=1}^{N} \left(y_i - \mathbf{w}^\top \mathbf{x}_i\right)^2$$

**Symbol definitions:**

- $\mathbf{w} \in \mathbb{R}^{294}$: the weight vector (one weight per feature)
- $\mathbf{x}_i \in \mathbb{R}^{294}$: the feature vector for case $i$
- $\mathbf{w}^\top \mathbf{x}_i$: the dot product = predicted duration
- $y_i$: the true duration
- $(y_i - \mathbf{w}^\top \mathbf{x}_i)^2$: the squared residual for case $i$

The optimal weights have a closed-form solution:

$$\hat{\mathbf{w}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$

**Numerical example** (toy, 2 features):

$\mathbf{X} = \begin{bmatrix} 1 & 45 \\ 1 & 120 \\ 1 & 200 \end{bmatrix}$ (intercept + age in minutes),
$\mathbf{y} = \begin{bmatrix} 60 \\ 110 \\ 180 \end{bmatrix}$

$$\mathbf{X}^\top \mathbf{X} = \begin{bmatrix} 3 & 365 \\ 365 & 53425 \end{bmatrix}, \qquad \mathbf{X}^\top \mathbf{y} = \begin{bmatrix} 350 \\ 47700 \end{bmatrix}$$

Solving $\hat{\mathbf{w}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$ gives
$\hat{w}_0 \approx 3.5$, $\hat{w}_1 \approx 0.87$ — meaning every 10-minute increase in
the scheduling window adds about 8.7 minutes of predicted duration.

**Result in project**: MAE ≈ 35–46 min depending on encoding. Worst overall — cannot capture non-linear interactions.

---

## 7.3 Ridge Regression

### Intuition

Ridge adds a **penalty** on the sum of squared weights to the OLS objective.
This prevents any single weight from growing too large, which stabilises predictions
when features are correlated (as text embedding dimensions often are).

**Objective:**

$$\mathcal{L}_{\text{Ridge}}(\boldsymbol{w}) = \sum_{i=1}^{N}\left(y_i - \mathbf{w}^\top \mathbf{x}_i\right)^2 + \alpha \sum_{j=1}^{p} w_j^2$$

**Symbol definitions:**

- $\alpha > 0$: the regularisation strength hyperparameter — larger $\alpha$ → smaller weights (more bias, less variance)
- $\sum_{j=1}^{p} w_j^2$: the L2 norm squared of the weight vector (also written $\|\mathbf{w}\|_2^2$)
- $p = 294$: number of features

**Term-by-term breakdown:**

1. First sum: the standard OLS fit quality term — minimise prediction error.
2. Second term: penalises large weights proportionally to their square. A weight of 10 is penalised 100 times as much as a weight of 1. This pushes all weights toward zero, but does not set any weight exactly to zero (unlike Lasso).

**Closed-form solution:**

$$\hat{\mathbf{w}}_{\text{Ridge}} = (\mathbf{X}^\top \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}$$

The $\alpha \mathbf{I}$ term adds a positive value to every diagonal of $\mathbf{X}^\top \mathbf{X}$,
guaranteeing the matrix is invertible (solving the multicollinearity problem).

**Result in project**: MAE ≈ 35–46 min — nearly identical to linear regression because the
feature space is well-conditioned after BERT embedding.

---

## 7.4 Lasso Regression

### Intuition

Lasso uses an L1 penalty ($|w_j|$) instead of Ridge's L2 penalty ($w_j^2$).
This creates a crucial difference: Lasso can set weights **exactly to zero**, performing
automatic feature selection.

**Objective:**

$$\mathcal{L}_{\text{Lasso}}(\boldsymbol{w}) = \sum_{i=1}^{N}\left(y_i - \mathbf{w}^\top \mathbf{x}_i\right)^2 + \alpha \sum_{j=1}^{p} |w_j|$$

**Why does L1 create sparsity while L2 does not?**

The L1 penalty forms a "diamond" constraint region in weight space; the OLS optimum can touch
this diamond at a corner where some coordinates are exactly zero.
The L2 penalty forms a "sphere" which has no corners — it always touches the OLS optimum at
a non-zero point.

```{.matplotlib}
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor('#0A0A0A')
for ax in (ax1, ax2):
    ax.set_facecolor('#0A0A0A')
    for spine in ax.spines.values(): spine.set_color('#444444')
    ax.tick_params(colors='#CCCCCC', labelsize=11)
    ax.set_aspect('equal')
    ax.axhline(0, color='#444444'); ax.axvline(0, color='#444444')

# L2 constraint region (circle)
theta = np.linspace(0, 2*np.pi, 300)
ax1.fill(np.cos(theta), np.sin(theta), color='#1565C0', alpha=0.5, label='L2 constraint')
ax1.plot(*[0.95, 0.95], 'o', color='#F57C00', markersize=12, label='OLS optimum')
# L2 solution — tangent point not on axis
ax1.plot(*[0.71, 0.71], 'D', color='#2E7D32', markersize=12, label='Ridge solution\n(no zero weights)')
ax1.set_title('Ridge (L2): smooth sphere', color='#CCCCCC', fontsize=11)
ax1.set_xlim(-1.5, 1.5); ax1.set_ylim(-1.5, 1.5)
ax1.legend(fontsize=9, facecolor='#1A1A1A', labelcolor='#CCCCCC')

# L1 constraint region (diamond)
diamond_x = [1, 0, -1, 0, 1]
diamond_y = [0, 1, 0, -1, 0]
ax2.fill(diamond_x, diamond_y, color='#C62828', alpha=0.5, label='L1 constraint')
ax2.plot(*[0.95, 0.95], 'o', color='#F57C00', markersize=12, label='OLS optimum')
# L1 solution — tangent at corner on axis
ax2.plot(*[1.0, 0.0], 'D', color='#2E7D32', markersize=12, label='Lasso solution\n(w₂ = 0 exactly)')
ax2.set_title('Lasso (L1): diamond with corners', color='#CCCCCC', fontsize=11)
ax2.set_xlim(-1.5, 1.5); ax2.set_ylim(-1.5, 1.5)
ax2.legend(fontsize=9, facecolor='#1A1A1A', labelcolor='#CCCCCC')

plt.tight_layout()
```

**What to observe**: The L2 constraint sphere has no corner for the OLS ellipse to snag on —
the optimal point has both weights non-zero.
The L1 diamond has corners on the axes; the OLS ellipse naturally touches at a corner,
forcing one weight to exactly zero (shown as a green diamond at the corner).

---

## 7.5 ElasticNet

### Intuition

ElasticNet combines both L1 and L2 penalties.
When features are correlated (like text embedding dimensions), pure Lasso tends to
arbitrarily pick one of several correlated features and zero out the rest.
ElasticNet uses L2 to keep correlated features together and L1 to still encourage sparsity.

**Objective:**

$$\mathcal{L}_{\text{EN}}(\boldsymbol{w}) = \sum_{i=1}^{N}(y_i - \mathbf{w}^\top \mathbf{x}_i)^2 + \alpha \left[\rho \sum_{j=1}^{p}|w_j| + \frac{1-\rho}{2} \sum_{j=1}^{p}w_j^2\right]$$

**Symbol definitions:**

- $\alpha > 0$: overall regularisation strength
- $\rho \in [0, 1]$: the L1 ratio — controls the mix between L1 and L2
  - $\rho = 1$: pure Lasso
  - $\rho = 0$: pure Ridge
  - $\rho = 0.5$: equal mix (typical default)

---

## 7.6 Random Forest

### Intuition

A random forest builds many decision trees, each on a random subset of the training data
and features, then averages their predictions.
No single tree is very accurate, but by averaging hundreds of decorrelated trees,
the random noise cancels out and the systematic prediction remains.

**Real-world analogy**: you want to estimate how long a road trip will take.
You ask 200 different people who each took the same route at different times (some on weekday
mornings, some on holiday evenings, some with different cars).
None of them give you the perfect answer, but their average estimate is much more reliable
than any one person's guess.

### Objective (Single Tree Node Split)

Each tree finds the best feature and split threshold to minimise prediction error in the
resulting subtrees.
The **mean squared error reduction** (impurity decrease) for a split at threshold $t$ on feature $j$:

$$\Delta_{\text{MSE}}(j, t) = \frac{n}{N}\,\text{MSE}(\mathcal{N}) - \frac{n_L}{N}\,\text{MSE}(\mathcal{N}_L) - \frac{n_R}{N}\,\text{MSE}(\mathcal{N}_R)$$

**Symbol definitions:**

- $\mathcal{N}$: the set of training samples at the current node
- $\mathcal{N}_L, \mathcal{N}_R$: the left and right child nodes after splitting
- $n, n_L, n_R$: the number of samples in each node
- $\text{MSE}(\mathcal{S}) = \frac{1}{|\mathcal{S}|}\sum_{i \in \mathcal{S}}(y_i - \bar{y}_\mathcal{S})^2$: the mean squared error within node $\mathcal{S}$, where $\bar{y}_\mathcal{S}$ is the mean duration
- $\Delta_{\text{MSE}}$: the reduction in MSE from this split — we choose the split maximising this

**Numerical example:**

Node with 4 cases: $y = [60, 80, 200, 220]$, $\bar{y} = 140$.

$$\text{MSE}(\mathcal{N}) = \frac{(60-140)^2 + (80-140)^2 + (200-140)^2 + (220-140)^2}{4}$$
$$= \frac{6400 + 3600 + 3600 + 6400}{4} = \frac{20000}{4} = 5000$$

Try split: left = $\{60, 80\}$, right = $\{200, 220\}$:

$$\text{MSE}(\mathcal{N}_L) = \frac{(60-70)^2 + (80-70)^2}{2} = \frac{100 + 100}{2} = 100$$

$$\text{MSE}(\mathcal{N}_R) = \frac{(200-210)^2 + (220-210)^2}{2} = \frac{100 + 100}{2} = 100$$

$$\Delta_{\text{MSE}} = \frac{4}{4}(5000) - \frac{2}{4}(100) - \frac{2}{4}(100) = 5000 - 50 - 50 = 4900$$

**Interpretation**: This split reduces MSE by 4900 — very effective.
The forest chooses splits that maximise this reduction, building trees that separate
short and long cases as cleanly as possible.

---

## 7.7 XGBoost

### Intuition

XGBoost (Extreme Gradient Boosting) builds trees **sequentially**.
Each new tree corrects the errors of all previous trees.
It is called "gradient boosting" because each tree fits the gradient of the loss function —
the direction in which predictions need to move to reduce error.

**Real-world analogy**: a team of specialists works in relay.
The first specialist makes their best prediction.
The second specialist looks only at where the first was wrong and tries to correct those errors.
The third specialist corrects the second's remaining errors, and so on.
Each specialist focuses entirely on the mistakes of the previous team.

### Objective Function (One Boosting Round)

At round $t$, we add a new tree $f_t(\mathbf{x})$ to the existing ensemble:

$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)$$

The tree is fit by minimising the **second-order Taylor expansion** of the loss:

$$\mathcal{L}^{(t)} \approx \sum_{i=1}^{N} \left[g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i)\right] + \Omega(f_t)$$

$$\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

**Symbol definitions:**

- $g_i = \frac{\partial \mathcal{L}(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}$: first derivative (gradient) of the loss with respect to the current prediction — tells us which direction to move prediction $i$
- $h_i = \frac{\partial^2 \mathcal{L}(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2}$: second derivative (Hessian) — tells us how fast the gradient is changing; controls the step size
- $f_t(\mathbf{x}_i)$: the new tree's prediction for case $i$ (an additive correction)
- $\Omega(f_t)$: regularisation on the new tree
- $T$: number of leaves in the new tree
- $\gamma$: minimum loss reduction required to make a split (pruning parameter)
- $\lambda$: L2 regularisation on the leaf weights
- $w_j$: the predicted value (weight) at leaf $j$

**For MSE loss** ($\mathcal{L} = \frac{1}{2}(y_i - \hat{y}_i)^2$):

$$g_i = \hat{y}_i^{(t-1)} - y_i \qquad (\text{the residual})$$
$$h_i = 1 \qquad (\text{constant for MSE})$$

The optimal leaf weight (closed form) for all samples in leaf $j$:

$$w_j^* = -\frac{\sum_{i \in \mathcal{I}_j} g_i}{\sum_{i \in \mathcal{I}_j} h_i + \lambda} = -\frac{\sum_{i \in \mathcal{I}_j}(\hat{y}_i - y_i)}{|\mathcal{I}_j| + \lambda}$$

**Interpretation**: The optimal leaf weight is the negative average residual in that leaf,
divided by a regularisation term.
A leaf with all under-predicted cases (positive residuals) gets a negative weight correction,
pulling predictions upward.

**Result in project**: XGBoost + TinySurgicalBERT: MAE = 26.38 ± 0.10 min — best result.

---

## 7.8 LightGBM

### Intuition

LightGBM is another gradient boosting framework with two key algorithmic improvements
over XGBoost that make it faster and often more accurate on datasets with many features:

1. **Leaf-wise growth**: standard trees grow level-by-level (every leaf splits);
   LightGBM grows only the single leaf with the largest loss reduction.
   This produces deeper, more asymmetric trees that can capture complex patterns with fewer
   total splits.

2. **Gradient-based one-sided sampling (GOSS)**: keeps all cases with large gradients
   (hard-to-predict cases) but randomly drops a fraction of easy cases.
   This speeds up training without losing much accuracy.

**Objective** (same as XGBoost but with `num_leaves` controlling tree depth):

$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i), \qquad \mathcal{L}^{(t)} \approx \sum_{i=1}^N\left[g_i f_t + \frac{1}{2}h_i f_t^2\right] + \Omega(f_t)$$

LightGBM adds early stopping: if validation loss has not improved for 20 consecutive rounds,
training stops automatically, preventing overfitting.

**Key hyperparameter**: `num_leaves` (15–127 in our search space) controls the maximum
number of leaves in a single tree.
Unlike XGBoost's `max_depth`, `num_leaves` directly controls model capacity independent of depth.

**Result in project**: LightGBM + TinySurgicalBERT: MAE = 26.43 ± 0.11 min — matches XGBoost.

---

## 7.9 MLP (Multi-Layer Perceptron)

### Intuition

An MLP is a neural network with one or more fully-connected hidden layers.
Unlike tree models that split the feature space into rectangular regions,
an MLP can learn smooth, non-linear transformations of any shape — in theory,
more expressive than trees.
In practice, MLPs are harder to train and require larger datasets for their advantage to materialise.

### Forward Pass

For a one-hidden-layer MLP:

$$\hat{y} = \mathbf{w}^{(2)\top} \sigma\!\left(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}\right) + b^{(2)}$$

**Symbol definitions:**

- $\mathbf{x} \in \mathbb{R}^{294}$: input feature vector
- $\mathbf{W}^{(1)} \in \mathbb{R}^{H \times 294}$: weight matrix of the hidden layer, $H$ = number of hidden units
- $\mathbf{b}^{(1)} \in \mathbb{R}^H$: bias vector of the hidden layer
- $\sigma(\cdot)$: the ReLU activation function, applied element-wise: $\sigma(z) = \max(0, z)$
- $\mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)} \in \mathbb{R}^H$: the pre-activation hidden layer values
- $\mathbf{w}^{(2)} \in \mathbb{R}^H$: weight vector of the output layer
- $b^{(2)} \in \mathbb{R}$: output layer bias
- $\hat{y} \in \mathbb{R}$: predicted duration

**Training objective** (MSE with L2 regularisation):

$$\mathcal{L}_{\text{MLP}} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2 + \alpha_{\text{MLP}} \sum_{\text{all weights}} w^2$$

**Numerical example** (1 hidden unit, 2 features):

$\mathbf{x} = [0.5, 0.3]$, $\mathbf{W}^{(1)} = [[1.2,\; -0.8]]$, $\mathbf{b}^{(1)} = [0.1]$

Pre-activation: $1.2(0.5) + (-0.8)(0.3) + 0.1 = 0.60 - 0.24 + 0.10 = 0.46$

After ReLU: $\sigma(0.46) = 0.46$ (positive, unchanged)

Output: $w^{(2)} = 180$, $b^{(2)} = 20$ → $\hat{y} = 180(0.46) + 20 = 82.8 + 20 = 102.8$ minutes.

**Result in project**: MLP + TinySurgicalBERT: MAE = 26.95 ± 0.29 min — competitive but higher variance.

---

## 7.10 Summary: All 8 Models Compared

```{.matplotlib}
import matplotlib.pyplot as plt
import numpy as np

models = ['Lin. Reg.', 'Ridge', 'Lasso', 'ElasticNet',
          'Rand.\nForest', 'XGBoost', 'LightGBM', 'MLP']
# TinySurgicalBERT MAE values from result.db
mae_tiny = [39.21, 39.17, 39.20, 39.19, 30.99, 26.38, 26.43, 26.95]
mae_struct = [46.66, 46.65, 46.65, 46.64, 36.93, 34.80, 34.79, 36.85]

x = np.arange(len(models))
w = 0.35

fig, ax = plt.subplots(figsize=(14, 4))
fig.patch.set_facecolor('#0A0A0A')
ax.set_facecolor('#0A0A0A')
for spine in ax.spines.values(): spine.set_color('#444444')
ax.tick_params(colors='#CCCCCC', labelsize=11)

bars1 = ax.bar(x - w/2, mae_tiny,   width=w, color='#2E7D32', label='TinySurgicalBERT', alpha=0.9, edgecolor='none')
bars2 = ax.bar(x + w/2, mae_struct, width=w, color='#6A1B9A', label='Structured Only',  alpha=0.9, edgecolor='none')

ax.axhline(26.38, color='#F57C00', lw=1.5, linestyle='--', alpha=0.7, label='Best: 26.38 min')
ax.set_xticks(x)
ax.set_xticklabels(models, color='#CCCCCC', fontsize=11)
ax.set_ylabel('MAE (minutes)', color='#CCCCCC')
ax.set_title('MAE by model: TinySurgicalBERT vs Structured Only', color='#CCCCCC', fontsize=12)
ax.legend(fontsize=11, facecolor='#1A1A1A', labelcolor='#CCCCCC')
ax.set_ylim(0, 55)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}', ha='center', color='#CCCCCC', fontsize=9)

plt.tight_layout()
```

**What to observe**: The gradient-boosted trees (XGBoost, LightGBM) achieve the lowest MAE
by a wide margin — approximately 12 minutes better than linear models and 4 minutes better
than Random Forest.
The gap between TinySurgicalBERT and Structured Only is largest for gradient-boosted trees
(~8 minutes), meaning gradient boosting is better at exploiting the semantic text features.
MLP is competitive with the tree models but shows higher variance (larger standard deviations
across folds).

| Model | MAE (TinySurgBERT) | MAE (Structured Only) | Benefit of text |
|---|---|---|---|
| Linear Regression | 39.21 | 46.66 | 7.45 min |
| Ridge | 39.17 | 46.65 | 7.48 min |
| Lasso | 39.20 | 46.65 | 7.45 min |
| ElasticNet | 39.19 | 46.64 | 7.45 min |
| Random Forest | 30.99 | 36.93 | 5.94 min |
| XGBoost | **26.38** | 34.80 | **8.42 min** |
| LightGBM | 26.43 | 34.79 | 8.36 min |
| MLP | 26.95 | 36.85 | 9.90 min |
