# TinySurgicalBERT — Section 6: Hyperparameter Optimisation with Optuna

## Introduction

Every machine learning model has **hyperparameters** — settings that are chosen before training
and are not learned from data (unlike model weights).
For example, the number of trees in XGBoost, the regularisation strength in Ridge regression,
or the number of neurons in an MLP.
Choosing good hyperparameters can be the difference between a mediocre model and the best
possible one for a given task.
This section explains Bayesian optimisation, the TPE algorithm used by Optuna, and exactly
how the search is structured in this project.

---

## 6.1 The Hyperparameter Problem

### What It Is

**Real-world analogy**: imagine you are tuning a radio receiver.
You have several dials: frequency, volume, treble, bass, antenna angle.
You could try every possible combination of dial positions — but there are billions of combinations.
A smarter approach is to start with random settings, listen to the result, turn a promising dial
slightly and listen again, and gradually home in on the best sound.
Bayesian hyperparameter optimisation is exactly this: a systematic strategy for exploring
the hyperparameter space that uses the results of past trials to decide which settings to try next.

### Grid Search vs. Random Search vs. Bayesian Optimisation

| Method | Strategy | Trials needed | Quality |
|---|---|---|---|
| Grid search | Try all combinations | Exponential in # params | Good if few params |
| Random search | Sample uniformly at random | Fixed budget | Often better than grid |
| Bayesian (TPE) | Build a model of good regions, sample from them | Fixed budget | Usually best |

With 8+ hyperparameters per model, grid search is computationally impossible.
Bayesian optimisation with 20 trials (as used in this project) typically finds settings
within 5–10% of the global optimum.

---

## 6.2 Bayesian Optimisation: Core Idea

### Intuition

Bayesian optimisation maintains a **surrogate model** — a cheap-to-evaluate statistical model
that approximates how good a given hyperparameter configuration is, based on results seen so far.
At each iteration, it uses this surrogate to pick the most promising next configuration
to evaluate, balancing two objectives:

- **Exploitation**: try configurations near the best ones found so far
- **Exploration**: try configurations in regions not yet evaluated, in case they are better

This balance prevents getting stuck in local optima and avoids wasting trials on obviously bad regions.

### The Acquisition Function

The acquisition function $a(\boldsymbol{\lambda})$ tells us how valuable it would be to
evaluate hyperparameter configuration $\boldsymbol{\lambda}$ next.
The next trial is:

$$\boldsymbol{\lambda}^* = \underset{\boldsymbol{\lambda}}{\arg\max}\; a(\boldsymbol{\lambda})$$

Common acquisition functions include Expected Improvement (EI), Upper Confidence Bound (UCB),
and the TPE approach used by Optuna.

---

## 6.3 TPE: Tree-Structured Parzen Estimators

### What It Is

TPE (Tree-Structured Parzen Estimator) is the specific algorithm used by Optuna.
Instead of fitting a single surrogate across the whole space, it fits two density models:

$$\boldsymbol{\lambda}^* = \underset{\boldsymbol{\lambda}}{\arg\max}\; \frac{\ell(\boldsymbol{\lambda})}{g(\boldsymbol{\lambda})}$$

**Symbol definitions:**

- $\boldsymbol{\lambda}$: a hyperparameter configuration (e.g., learning_rate=0.05, n_estimators=300)
- $\ell(\boldsymbol{\lambda})$: a density model trained on the **good** configurations — those with objective value below the $\gamma$-th quantile (top fraction)
- $g(\boldsymbol{\lambda})$: a density model trained on the **bad** configurations — those above the $\gamma$-th quantile
- $\gamma$: the quantile split (Optuna default: 0.25 — top 25% of trials are "good")
- $\frac{\ell(\boldsymbol{\lambda})}{g(\boldsymbol{\lambda})}$: the TPE acquisition ratio — configurations that are likely under the good model but unlikely under the bad model are most worth trying

### Why This Works

Maximising $\ell / g$ finds configurations that:
1. Are probable among the good runs (exploitation — stay in promising regions)
2. Are improbable among the bad runs (exploration — avoid bad regions)

If a configuration is possible in the good region but never appears in the bad region,
$\ell(\boldsymbol{\lambda}) / g(\boldsymbol{\lambda})$ is large → this configuration is highly
recommended.

### Density Estimation

Both $\ell(\boldsymbol{\lambda})$ and $g(\boldsymbol{\lambda})$ are estimated using
**Parzen windows** (kernel density estimation with Gaussian kernels):

$$\ell(\boldsymbol{\lambda}) = \frac{1}{n_\ell} \sum_{i \in \text{good}} K\!\left(\boldsymbol{\lambda}, \boldsymbol{\lambda}_i, h\right)$$

$$K(\boldsymbol{\lambda}, \boldsymbol{\lambda}_i, h) = \frac{1}{\sqrt{2\pi} h} \exp\!\left(-\frac{\|\boldsymbol{\lambda} - \boldsymbol{\lambda}_i\|^2}{2h^2}\right)$$

**Symbol definitions:**

- $n_\ell$: number of good configurations (top $\gamma$ fraction of all trials)
- $\boldsymbol{\lambda}_i$: the $i$-th observed good configuration
- $K(\cdot)$: a Gaussian kernel — a "bump" centred at $\boldsymbol{\lambda}_i$ with bandwidth $h$
- $h$: bandwidth parameter controlling how wide each bump is (Optuna uses adaptive bandwidth)

**Intuition**: $\ell(\boldsymbol{\lambda})$ is high wherever good configurations cluster.
A new configuration is promising if it lands in a region that is dense with good results.

---

## 6.4 The Optuna Search Spaces in This Project

### Per-Model Hyperparameter Bounds

For each of the 8 downstream models, a different search space was defined:

**Linear models (Ridge, Lasso, ElasticNet):**

| Hyperparameter | Range | Scale |
|---|---|---|
| Ridge $\alpha$ (regularisation) | $[10^{-3},\; 100]$ | Log-uniform |
| Lasso $\alpha$ | $[10^{-3},\; 100]$ | Log-uniform |
| ElasticNet $\alpha$ | $[10^{-3},\; 100]$ | Log-uniform |
| ElasticNet $\ell_1$-ratio | $[0.0,\; 1.0]$ | Uniform |

**Random Forest:**

| Hyperparameter | Range | Scale |
|---|---|---|
| n_estimators | $[50,\; 250]$ | Integer uniform |
| max_depth | $[3,\; 10]$ | Integer uniform |
| min_samples_split | $[2,\; 20]$ | Integer uniform |
| max_features | $\{$`"sqrt"`, `"log2"`$\}$ | Categorical |

**XGBoost:**

| Hyperparameter | Range | Scale |
|---|---|---|
| n_estimators | $[100,\; 500]$ | Integer uniform |
| learning_rate | $[0.01,\; 0.3]$ | Log-uniform |
| max_depth | $[3,\; 8]$ | Integer uniform |
| subsample | $[0.6,\; 1.0]$ | Uniform |
| colsample_bytree | $[0.6,\; 1.0]$ | Uniform |
| reg_alpha (L1) | $[10^{-4},\; 10]$ | Log-uniform |
| reg_lambda (L2) | $[10^{-4},\; 10]$ | Log-uniform |

**LightGBM** (same structure as XGBoost, with additional `num_leaves` $\in [15, 127]$)

**MLP:**

| Hyperparameter | Range | Scale |
|---|---|---|
| hidden layer sizes | $\{(128,),\; (256,),\; (128,128)\}$ | Categorical |
| learning_rate_init | $[10^{-4},\; 10^{-1}]$ | Log-uniform |
| alpha (L2) | $[10^{-5},\; 10^{-1}]$ | Log-uniform |

### Log-Uniform Sampling

For hyperparameters that span several orders of magnitude (e.g., regularisation from 0.001 to 100),
sampling uniformly would spend 99.9% of trials on values above 0.1, missing the important
low end.
Log-uniform sampling fixes this by sampling $\log(\alpha)$ uniformly, then exponentiating:

$$\alpha = e^{u}, \quad u \sim \text{Uniform}(\log(\alpha_{\min}),\; \log(\alpha_{\max}))$$

```{.matplotlib}
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)
n = 1000

# Uniform vs log-uniform sampling for alpha in [0.001, 100]
alpha_min, alpha_max = 0.001, 100.0

uniform_samples    = rng.uniform(alpha_min, alpha_max, n)
log_uniform_samples = np.exp(rng.uniform(np.log(alpha_min), np.log(alpha_max), n))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor('#0A0A0A')

for ax in (ax1, ax2):
    ax.set_facecolor('#0A0A0A')
    for spine in ax.spines.values(): spine.set_color('#444444')
    ax.tick_params(colors='#CCCCCC', labelsize=11)
    ax.xaxis.label.set_color('#CCCCCC')
    ax.yaxis.label.set_color('#CCCCCC')

ax1.hist(uniform_samples, bins=60, color='#C62828', edgecolor='none', alpha=0.85)
ax1.set_xlabel('Sampled alpha value')
ax1.set_ylabel('Count')
ax1.set_title('Uniform sampling (misses small values)', color='#CCCCCC', fontsize=11)

ax2.hist(log_uniform_samples, bins=np.logspace(np.log10(alpha_min), np.log10(alpha_max), 60),
         color='#2E7D32', edgecolor='none', alpha=0.85)
ax2.set_xscale('log')
ax2.set_xlabel('Sampled alpha value (log scale)')
ax2.set_ylabel('Count')
ax2.set_title('Log-uniform sampling (covers all scales equally)', color='#CCCCCC', fontsize=11)

plt.tight_layout()
```

**What to observe:**

- **Left panel**: Uniform sampling clusters almost all 1000 samples above 10 — the small values
  (0.001 to 10) receive almost no exploration, even though they are often the most important region.
- **Right panel**: Log-uniform sampling distributes samples evenly across each order of magnitude.
  The range 0.001–0.01 gets the same number of samples as 10–100.
  This is why log-uniform is always used for regularisation and learning rate hyperparameters.

---

## 6.5 Numerical Example: One Optuna Trial for XGBoost

Suppose we are on trial 8 of 20 for XGBoost.
Trials 1–7 have been evaluated; the best so far used `learning_rate=0.05, n_estimators=300`.

TPE fits $\ell(\boldsymbol{\lambda})$ on the top 25% of 7 trials (top 2 = the 2 best).
For `learning_rate` alone, the good observations are $\{0.05, 0.06\}$.

A Gaussian kernel centred on each:

$$\ell(\text{lr}) \propto K(\text{lr},\; 0.05,\; h) + K(\text{lr},\; 0.06,\; h)$$

This density peaks near $\text{lr} = 0.055$, suggesting the next trial should sample
`learning_rate` close to 0.055.
Meanwhile $g(\text{lr})$ peaks elsewhere (based on the 5 worse trials), so the ratio
$\ell / g$ is maximised near 0.055.

Optuna samples `learning_rate = 0.053` for trial 8, trains XGBoost with that learning rate,
evaluates MAE on the fold's validation set, and records the result.
This process repeats 20 times, and the configuration with the lowest MAE is selected.

**Interpretation**: By trial 20, Optuna has effectively searched the most promising region
of the learning rate space (0.03–0.08) far more thoroughly than random search would,
while still occasionally trying extreme values (very small or very large) to avoid missing
a counterintuitive optimum.

---

## 6.6 Code: Optuna Integration

```python
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress verbose output

def objective_xgb(trial, X_train, y_train, X_val, y_val):
    """Objective function: Optuna minimises the value this returns."""
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.30, log=True),
        'max_depth':         trial.suggest_int('max_depth', 3, 8),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'tree_method': 'hist',   # GPU-accelerated histogram method
        'device':      'cuda',   # use GPU if available
        'random_state':  42,
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_val)
    return mean_absolute_error(y_val, preds)   # Optuna minimises this

# Run 20 trials (budget set for speed; more trials = better result)
study = optuna.create_study(
    direction='minimize',                          # we want lower MAE
    sampler=optuna.samplers.TPESampler(
        n_startup_trials=5,                        # first 5 trials: random exploration
        seed=42
    )
)
study.optimize(
    lambda trial: objective_xgb(trial, X_train, y_train, X_val, y_val),
    n_trials=20
)

best_params = study.best_params
print(f"Best MAE: {study.best_value:.4f}")
# Output: Best MAE: 26.3812 (varies by fold)
print(f"Best params: {best_params}")
# Output: {'n_estimators': 347, 'learning_rate': 0.0512, 'max_depth': 6, ...}
```

---

## Summary

| Concept | Key Takeaway |
|---|---|
| Hyperparameters | Settings chosen before training; not learned from data |
| Grid search | Infeasible for 7+ hyperparameters |
| Random search | Better, but ignores results of past trials |
| Bayesian optimisation | Uses past results to guide where to search next |
| TPE | Fits separate density models on good vs. bad configs; maximises their ratio |
| Log-uniform sampling | Necessary for regularisation and learning rate to cover all scales equally |
| n_startup_trials=5 | First 5 trials are random (pure exploration) before TPE takes over |
| n_trials=20 | Budget used in this project; 20 trials per model per fold |
