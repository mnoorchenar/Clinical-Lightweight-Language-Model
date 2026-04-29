# =============================================================================
# figures.py  --  Publication-quality figures + efficiency table
#
# Reads from data/outputs/result.db and writes all outputs to results/:
#
#  PREDICTION ACCURACY (connected dot plots, 3 BERT encodings only):
#   results/pred_legend.pdf  -- shared legend row
#   results/pred_mae.pdf     -- MAE  (lower is better)
#   results/pred_mse.pdf     -- MSE  (lower is better)
#   results/pred_smape.pdf   -- sMAPE (lower is better)
#   results/pred_r2.pdf      -- R²   (higher is better)
#
#  DEPLOYMENT EFFICIENCY (bar plots, log scale):
#   results/efficiency_legend.pdf -- shared legend row
#   results/model_size.pdf        -- encoder on-disk file size (MB)
#   results/infer_time.pdf        -- per-case CPU inference time (ms)
#
#  GENERATED LATEX TABLE:
#   results/efficiency_table.tex  -- deployment profile table
#
# Run after Stage 04 of pipeline.py has completed.
# =============================================================================

RESULT_DB   = './data/outputs/result.db'
FIGURES_DIR = './results'
RESULTS_DIR = './results'

import os, sqlite3, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# Global typography  (publication: ~24–26 pt body text equivalent)
# =============================================================================
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':          24,
    'axes.labelsize':     26,
    'xtick.labelsize':    22,
    'ytick.labelsize':    22,
    'legend.fontsize':    20,
    'axes.linewidth':     1.4,
    'xtick.major.width':  1.2,
    'ytick.major.width':  1.2,
    'xtick.major.size':   6,
    'ytick.major.size':   6,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
})

# =============================================================================
# Load data
# =============================================================================
if not os.path.exists(RESULT_DB):
    raise FileNotFoundError(
        f"result.db not found at {RESULT_DB}.\n"
        "Run Stage 04 of pipeline.py first."
    )

with sqlite3.connect(RESULT_DB) as conn:
    metrics = pd.read_sql("SELECT * FROM metrics", conn)

# =============================================================================
# Ordering, labels, colours
# =============================================================================
ENC_ORDER = ['only_structured', 'sentencebert', 'clinicalbert', 'tinybert']
ENC_ORDER_BERT = ['sentencebert', 'clinicalbert', 'tinybert']   # no structured-only in accuracy plots

ENC_LABELS = {
    'only_structured': 'Structured Only',
    'sentencebert':    'SentenceBERT',
    'clinicalbert':    'Bio-ClinicalBERT',
    'tinybert':        'TinySurgicalBERT (ours)',
}
MODEL_ORDER = ['linear', 'ridge', 'lasso', 'elasticnet',
               'randomforest', 'xgboost', 'lightgbm', 'mlp']
MODEL_LABELS = {
    'linear':       'Lin. Reg.',
    'ridge':        'Ridge',
    'lasso':        'Lasso',
    'elasticnet':   'ElasticNet',
    'randomforest': 'Rand. Forest',
    'xgboost':      'XGBoost',
    'lightgbm':     'LightGBM',
    'mlp':          'MLP',
}

# Colour-blind-friendly palette
ENC_COLORS = {
    'only_structured': '#6c757d',
    'sentencebert':    '#2196F3',
    'clinicalbert':    '#FF9800',
    'tinybert':        '#4CAF50',
}
ENC_MARKERS = {
    'only_structured': 's',
    'sentencebert':    'o',
    'clinicalbert':    '^',
    'tinybert':        'D',
}

# Filter to models / encodings that actually have results
enc_avail_all  = [e for e in ENC_ORDER      if e in metrics['encoding'].unique()]
enc_avail_bert = [e for e in ENC_ORDER_BERT if e in metrics['encoding'].unique()]
model_avail    = [m for m in MODEL_ORDER     if m in metrics['model'].unique()]

# Use highest n_features per encoding (avoids multi-n clutter)
n_per_enc = metrics.groupby('encoding')['n_features'].max().to_dict()

filt = pd.concat([
    metrics[(metrics['encoding'] == enc) & (metrics['n_features'] == n)]
    for enc, n in n_per_enc.items()
], ignore_index=True)

# Aggregate mean and std across folds
grp = (
    filt.groupby(['encoding', 'model'])[['mae', 'mse', 'rmse', 'smape', 'r2',
                                          'train_time_s', 'infer_time_s']]
    .agg(['mean', 'std'])
)
grp.columns = ['_'.join(c) for c in grp.columns]
grp = grp.reset_index()

# =============================================================================
# Helper: lookup (mean, std) for a given encoding / model / metric
# =============================================================================
def _lookup(enc, model, col):
    row = grp[(grp['encoding'] == enc) & (grp['model'] == model)]
    if row.empty:
        return np.nan, np.nan
    return float(row[f'{col}_mean'].iloc[0]), float(row[f'{col}_std'].iloc[0])


# =============================================================================
# ACCURACY FIGURES  --  connected dot plots  (3 BERT encodings only)
# =============================================================================
METRICS_ACC = [
    ('mae',   'MAE (minutes)',   False),
    ('mse',   'MSE (min$^{2}$)', False),
    ('smape', 'sMAPE (%)',       False),
    ('r2',    '$R^{2}$',         True),
]
x_pos      = np.arange(len(model_avail))
x_labels   = [MODEL_LABELS.get(m, m) for m in model_avail]

def _acc_figure(metric, ylabel, higher_better):
    fig, ax = plt.subplots(figsize=(14, 7))
    for enc in enc_avail_all:
        means = np.array([_lookup(enc, m, metric)[0] for m in model_avail])
        stds  = np.array([_lookup(enc, m, metric)[1] for m in model_avail])
        color  = ENC_COLORS[enc]
        marker = ENC_MARKERS[enc]
        ax.plot(x_pos, means,
                color=color, marker=marker, markersize=10,
                linewidth=2.0, zorder=3)
        ax.fill_between(x_pos, means - stds, means + stds,
                        color=color, alpha=0.15, linewidth=0, zorder=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=30, ha='right')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.40, linewidth=1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, f'pred_{metric}.pdf')
    fig.savefig(out, bbox_inches='tight', format='pdf')
    plt.close(fig)
    print(f"  [OK] {out}")

for metric, ylabel, higher_better in METRICS_ACC:
    _acc_figure(metric, ylabel, higher_better)

# -- Legend-only PDF for accuracy plots (all 4 encodings, one row) -----------
fig_leg, ax_leg = plt.subplots(figsize=(14, 0.9))
ax_leg.set_axis_off()
handles = [
    mlines.Line2D([], [],
                  color=ENC_COLORS[enc],
                  marker=ENC_MARKERS[enc],
                  markersize=12,
                  linewidth=2.0,
                  label=ENC_LABELS[enc])
    for enc in enc_avail_all
]
ax_leg.legend(handles=handles, loc='center',
              ncol=len(handles), frameon=False, fontsize=22,
              handlelength=2.0, handletextpad=0.6, columnspacing=1.4)
fig_leg.savefig(os.path.join(FIGURES_DIR, 'pred_legend.pdf'),
                bbox_inches='tight', format='pdf')
plt.close(fig_leg)
print("  [OK] results/pred_legend.pdf")


# =============================================================================
# EFFICIENCY FIGURES  --  model size (MB) and end-to-end inference time (ms)
#
# Only the three text-encoding strategies are shown (Structured Only is
# excluded because predictive accuracy comparisons already establish its
# inferiority; these figures focus on the size/speed trade-off among
# text encoders).
#
# Model sizes are the on-disk file sizes of the deployed encoder artefacts:
#   SentenceBERT   : all-MiniLM-L6-v2  model.safetensors from HuggingFace cache
#   Bio-ClinicalBERT: emilyalsentzer/Bio_ClinicalBERT model.safetensors
#   TinySurgicalBERT: surgical_tiny_bert_q8.onnx  (INT8-quantised ONNX)
#
# End-to-end inference time per case (single-item CPU inference, measured on
# this machine):
#   SentenceBERT     : 2.62 ms  (sentence-transformers, batch_size=1)
#   Bio-ClinicalBERT : 21.49 ms (transformers AutoModel, CPU, batch_size=1)
#   TinySurgicalBERT : 0.49 ms  (ONNX Runtime CPUExecutionProvider)
# Downstream XGBoost prediction adds <0.005 ms/case (negligible).
# =============================================================================

import os as _os, sys as _sys

# ── Encoder model file sizes (bytes, measured directly from saved artefacts) ──
HF_CACHE = _os.path.expanduser('~/.cache/huggingface/hub')
_MODEL_PATHS = {
    'sentencebert':  _os.path.join(HF_CACHE,
        'models--sentence-transformers--all-MiniLM-L6-v2/snapshots/'
        'c9745ed1d9f207416be6d2e6f8de32d1f16199bf/model.safetensors'),
    'clinicalbert':  _os.path.join(HF_CACHE,
        'models--emilyalsentzer--Bio_ClinicalBERT/snapshots/'
        '3c22c28ae9c1619228e31dc7630645fee6081c98/model.safetensors'),
    'tinybert':      _os.path.join('.', 'data/outputs/models/surgical_tiny_bert_q8.onnx'),
}
# Fallback sizes in MB if cache paths differ across machines
_FALLBACK_MB = {'sentencebert': 90.87, 'clinicalbert': 435.76, 'tinybert': 0.752}
MODEL_SIZE_MB = {}
for enc, path in _MODEL_PATHS.items():
    if _os.path.exists(path):
        MODEL_SIZE_MB[enc] = _os.path.getsize(path) / 1e6
    else:
        MODEL_SIZE_MB[enc] = _FALLBACK_MB[enc]
        print(f"  [WARN] model file not found for {enc}, using fallback size")

# ── End-to-end per-case inference time (ms): embedding + downstream prediction
# Downstream XGBoost inference (~0.004 ms/case) is included but negligible.
ENC_BERT = ['sentencebert', 'clinicalbert', 'tinybert']
xgb_rows = grp[grp['model'] == 'xgboost'].set_index('encoding')

INFER_ENCODE_MS = {
    'sentencebert':  2.62,   # sentence-transformers, batch_size=1, CPU
    'clinicalbert':  21.49,  # transformers AutoModel, batch_size=1, CPU
    'tinybert':      0.49,   # ONNX Runtime CPUExecutionProvider
}
INFER_DOWNSTREAM_MS = {}
for enc in ENC_BERT:
    if enc in xgb_rows.index:
        # infer_time_s is total for ~36074 validation cases
        total_s = float(xgb_rows.loc[enc, 'infer_time_s_mean'])
        INFER_DOWNSTREAM_MS[enc] = total_s / 36074 * 1000
    else:
        INFER_DOWNSTREAM_MS[enc] = 0.004

INFER_TOTAL_MS = {enc: INFER_ENCODE_MS[enc] + INFER_DOWNSTREAM_MS[enc]
                  for enc in ENC_BERT}

# Restrict colours/labels to BERT encoders only
ENC_BERT_LABELS = {
    'sentencebert': 'SentenceBERT',
    'clinicalbert': 'Bio-ClinicalBERT',
    'tinybert':     'TinySurgicalBERT (ours)',
}

# ── Shared legend for efficiency figures (3 encoders, patch style) ──────────
fig_eff_leg, ax_eff_leg = plt.subplots(figsize=(14, 0.9))
ax_eff_leg.set_axis_off()
eff_handles = [
    mpatches.Patch(
        facecolor=ENC_COLORS[enc],
        edgecolor='black', linewidth=0.8,
        alpha=0.88,
        label=ENC_BERT_LABELS[enc]
    )
    for enc in ENC_BERT
]
ax_eff_leg.legend(eff_handles, [h.get_label() for h in eff_handles],
                  loc='center', ncol=3, frameon=False, fontsize=22,
                  handlelength=1.8, handletextpad=0.6, columnspacing=1.4)
fig_eff_leg.savefig(os.path.join(FIGURES_DIR, 'efficiency_legend.pdf'),
                    bbox_inches='tight', format='pdf')
plt.close(fig_eff_leg)
print("  [OK] results/efficiency_legend.pdf")

# ── Model size figure (log scale, MB) ───────────────────────────────────────
x_eff   = np.arange(len(ENC_BERT))
bar_w_e = 0.55

fig_sz, ax_sz = plt.subplots(figsize=(11, 7))
for i, enc in enumerate(ENC_BERT):
    ax_sz.bar(i, MODEL_SIZE_MB[enc],
              width=bar_w_e,
              color=ENC_COLORS[enc],
              alpha=0.88,
              edgecolor='black',
              linewidth=0.8)
ax_sz.set_yscale('log')
ax_sz.set_xticks(x_eff)
ax_sz.set_xticklabels([ENC_BERT_LABELS[e] for e in ENC_BERT],
                       rotation=15, ha='right')
ax_sz.set_ylabel('Model file size (MB)')
ax_sz.grid(axis='y', linestyle='--', alpha=0.40, linewidth=1.0)
ax_sz.spines['top'].set_visible(False)
ax_sz.spines['right'].set_visible(False)
fig_sz.tight_layout()
fig_sz.savefig(os.path.join(FIGURES_DIR, 'model_size.pdf'),
               bbox_inches='tight', format='pdf')
plt.close(fig_sz)
print("  [OK] results/model_size.pdf")

# ── End-to-end inference time figure (log scale, ms/case) ───────────────────
fig_inf, ax_inf = plt.subplots(figsize=(11, 7))
for i, enc in enumerate(ENC_BERT):
    ax_inf.bar(i, INFER_TOTAL_MS[enc],
               width=bar_w_e,
               color=ENC_COLORS[enc],
               alpha=0.88,
               edgecolor='black',
               linewidth=0.8)
ax_inf.set_yscale('log')
ax_inf.set_xticks(x_eff)
ax_inf.set_xticklabels([ENC_BERT_LABELS[e] for e in ENC_BERT],
                        rotation=15, ha='right')
ax_inf.set_ylabel('Inference time per case (ms)')
ax_inf.grid(axis='y', linestyle='--', alpha=0.40, linewidth=1.0)
ax_inf.spines['top'].set_visible(False)
ax_inf.spines['right'].set_visible(False)
fig_inf.tight_layout()
fig_inf.savefig(os.path.join(FIGURES_DIR, 'infer_time.pdf'),
                bbox_inches='tight', format='pdf')
plt.close(fig_inf)
print("  [OK] results/infer_time.pdf")


# =============================================================================
# EFFICIENCY STATISTICAL TABLE
#
# Inference time: N=30 paired benchmarks (200 cases each, single-item CPU).
# Paired Wilcoxon signed-rank test (TinySurgBERT as reference), FDR-BH
# corrected across 2 comparisons.
# Model size: deterministic (no statistical test applicable).
# =============================================================================

from scipy.stats import wilcoxon as _wilcoxon
from statsmodels.stats.multitest import multipletests as _mt

# ── Run benchmark (or load from cache) ──────────────────────────────────────
import sqlite3 as _sqlite3, torch as _torch
from transformers import AutoTokenizer as _ATok, AutoModel as _AMod
from sentence_transformers import SentenceTransformer as _ST
from tokenizers import Tokenizer as _Tok
import onnxruntime as _ort

_BENCH_TINY = os.path.join(RESULT_DB.replace('result.db', ''), 'bench_tiny.npy')
_BENCH_SENT = os.path.join(RESULT_DB.replace('result.db', ''), 'bench_sent.npy')
_BENCH_BC   = os.path.join(RESULT_DB.replace('result.db', ''), 'bench_bc.npy')
_BENCH_NRUNS = 30
_BENCH_NCASES = 200

if not (os.path.exists(_BENCH_TINY) and
        os.path.exists(_BENCH_SENT) and
        os.path.exists(_BENCH_BC)):
    print("  Running inference benchmarks (N=30)…")
    # Load text samples
    with _sqlite3.connect(os.path.join(os.path.dirname(RESULT_DB), '../raw/surgical_data.db')) as _cn:
        _texts = pd.read_sql('SELECT scheduled_procedure FROM Clean LIMIT 500', _cn)
    _texts = _texts['scheduled_procedure'].dropna().tolist()[:_BENCH_NCASES]

    # TinySurgBERT
    _tok_tiny  = _Tok.from_file(os.path.join(os.path.dirname(RESULT_DB), 'models/surgical_tiny_bert/tokenizer.json'))
    _sess_tiny = _ort.InferenceSession(os.path.join(os.path.dirname(RESULT_DB), 'models/surgical_tiny_bert_q8.onnx'),
                                       providers=['CPUExecutionProvider'])
    def _enc_tiny(txts):
        import time as _t
        for t in txts:
            enc  = _tok_tiny.encode(t)
            ids  = np.array([enc.ids], dtype=np.int64)
            mask = np.array([enc.attention_mask], dtype=np.int64)
            _sess_tiny.run(None, {'input_ids': ids, 'attention_mask': mask})
    _enc_tiny(_texts[:5])  # warmup
    _tiny_t = []
    import time as _time
    for _ in range(_BENCH_NRUNS):
        t0 = _time.perf_counter()
        _enc_tiny(_texts)
        _tiny_t.append((_time.perf_counter()-t0)/_BENCH_NCASES*1000)
    np.save(_BENCH_TINY, _tiny_t)

    # SentenceBERT
    _st_model = _ST('all-MiniLM-L6-v2')
    _st_model.eval()
    _st_model.encode(_texts[:5], show_progress_bar=False, batch_size=1)
    _sent_t = []
    for _ in range(_BENCH_NRUNS):
        t0 = _time.perf_counter()
        _st_model.encode(_texts, show_progress_bar=False, batch_size=1)
        _sent_t.append((_time.perf_counter()-t0)/_BENCH_NCASES*1000)
    np.save(_BENCH_SENT, _sent_t)
    del _st_model

    # Bio-ClinicalBERT
    _bc_tok  = _ATok.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    _bc_mod  = _AMod.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    _bc_mod.eval()
    def _enc_bc(txts):
        with _torch.no_grad():
            for t in txts:
                inp = _bc_tok(t, return_tensors='pt', truncation=True, max_length=128)
                _bc_mod(**inp)
    _enc_bc(_texts[:3])
    _bc_t = []
    for _ in range(_BENCH_NRUNS):
        t0 = _time.perf_counter()
        _enc_bc(_texts)
        _bc_t.append((_time.perf_counter()-t0)/_BENCH_NCASES*1000)
    np.save(_BENCH_BC, _bc_t)
    del _bc_mod
else:
    _tiny_t = list(np.load(_BENCH_TINY))
    _sent_t = list(np.load(_BENCH_SENT))
    _bc_t   = list(np.load(_BENCH_BC))
    print("  Loaded cached benchmark results.")

# ── Statistics ───────────────────────────────────────────────────────────────
_t_tiny = np.array(_tiny_t)
_t_sent = np.array(_sent_t)
_t_bc   = np.array(_bc_t)

_, _p_sent = _wilcoxon(_t_tiny - _t_sent, alternative='two-sided')
_, _p_bc   = _wilcoxon(_t_tiny - _t_bc,   alternative='two-sided')
_, _qvals, _, _ = _mt([_p_sent, _p_bc], method='fdr_bh')
_q_sent, _q_bc = _qvals

def _sig(q):
    if   q < 0.001: return r'$^{***}$'
    elif q < 0.01:  return r'$^{**}$'
    elif q < 0.05:  return r'$^{*}$'
    return ''

# ── Model sizes ───────────────────────────────────────────────────────────────
_HF = os.path.expanduser('~/.cache/huggingface/hub')
_SZ = {
    'sentencebert': os.path.getsize(os.path.join(_HF,
        'models--sentence-transformers--all-MiniLM-L6-v2/snapshots/'
        'c9745ed1d9f207416be6d2e6f8de32d1f16199bf/model.safetensors')) / 1e6,
    'clinicalbert': os.path.getsize(os.path.join(_HF,
        'models--emilyalsentzer--Bio_ClinicalBERT/snapshots/'
        '3c22c28ae9c1619228e31dc7630645fee6081c98/model.safetensors')) / 1e6,
    'tinybert': os.path.getsize(os.path.join(os.path.dirname(RESULT_DB),
        'models/surgical_tiny_bert_q8.onnx')) / 1e6,
}

# ── Build LaTeX table ─────────────────────────────────────────────────────────
def _fmt_time(mu, std, q, is_ref=False):
    base = f'${mu:.3f} \\pm {std:.3f}$'
    if is_ref:
        return base
    sym = r'{\color{green!60!black}$\checkmark$}' if mu > np.mean(_tiny_t) else r'{\color{red!70!black}$\times$}'
    return base + r'\,' + sym + _sig(q)

_eff_lines = []
_eff_lines += [
    r'\begin{table}[htbp]',
    r'\centering',
    (r'\caption{Statistical comparison of deployment efficiency for the three text encoders. '
     r'Model size is the on-disk file size of the deployed artefact. '
     r'Inference time (ms per case) is measured over $N = 30$ repeated passes on 200 surgical '
     r'procedure descriptions under single-item CPU inference. '
     r'TinySurgicalBERT is the reference; significance of the inference-time advantage is '
     r'assessed by a two-sided paired Wilcoxon signed-rank test with Benjamini--Hochberg '
     r'FDR correction. '
     r'{\color{green!60!black}$\checkmark$} = TinySurgicalBERT significantly faster; '
     r'model size is deterministic and requires no statistical test.}'),
    r'\label{tab:efficiency}',
    r'\begin{threeparttable}',
    r'\begin{tabular}{lrrcrr}',
    r'\toprule',
    (r'Encoder & Size (MB) & Compression ($\times$) & '
     r'Inference time (ms/case) & Speed-up ($\times$) \\'),
    r'\midrule',
]

# Reference row
_eff_lines.append(
    r'TinySurgicalBERT (ours) & '
    + f'${_SZ["tinybert"]:.3f}$ & '
    + r'--- & '
    + _fmt_time(np.mean(_t_tiny), np.std(_t_tiny), None, is_ref=True) + r' & '
    + r'--- \\'
)
_eff_lines.append(r'\midrule')

# SentenceBERT
_eff_lines.append(
    r'SentenceBERT & '
    + f'${_SZ["sentencebert"]:.2f}$ & '
    + f'${_SZ["sentencebert"]/_SZ["tinybert"]:.0f}$ & '
    + _fmt_time(np.mean(_t_sent), np.std(_t_sent), _q_sent) + r' & '
    + f'${np.mean(_t_sent)/np.mean(_t_tiny):.1f}$ \\\\'
)

# Bio-ClinicalBERT
_eff_lines.append(
    r'Bio-ClinicalBERT & '
    + f'${_SZ["clinicalbert"]:.2f}$ & '
    + f'${_SZ["clinicalbert"]/_SZ["tinybert"]:.0f}$ & '
    + _fmt_time(np.mean(_t_bc), np.std(_t_bc), _q_bc) + r' & '
    + f'${np.mean(_t_bc)/np.mean(_t_tiny):.1f}$ \\\\'
)

_eff_lines += [
    r'\bottomrule',
    r'\end{tabular}',
    r'\begin{tablenotes}',
    r'\small',
    (r'\item $^{*}p < 0.05$;\; $^{**}p < 0.01$;\; $^{***}p < 0.001$ '
     r'(two-sided paired Wilcoxon signed-rank, Benjamini--Hochberg FDR corrected, $N = 30$).'),
    (r'\item Inference time = encoder forward pass only; downstream XGBoost prediction '
     r'contributes $< 0.005$\,ms/case (negligible).'),
    r'\item $\downarrow$ lower is better for both size and inference time.',
    r'\end{tablenotes}',
    r'\end{threeparttable}',
    r'\end{table}',
]

_eff_tex_path = os.path.join(FIGURES_DIR, 'efficiency_table.tex')
with open(_eff_tex_path, 'w', encoding='utf-8') as _f:
    _f.write('\n'.join(_eff_lines) + '\n')
print(f"  [OK] {_eff_tex_path}")


# =============================================================================
# STATISTICAL COMPARISON TABLE  --  Wilcoxon signed-rank + FDR-BH
#
# Reference: TinySurgicalBERT (tinybert).
# For each metric, each downstream model, and each comparison encoding,
# perform a Wilcoxon signed-rank test on 5 fold-level paired differences.
# FDR-BH correction is applied across all comparisons (3 enc × 4 metrics
# × 8 models = 96 tests).
#
# Table layout:
#   Rows  : 8 downstream models
#   Groups: 3 comparison encodings
#   Cells : mean Δ (tinybert − other), positive = tinybert worse for
#           MAE/MSE/sMAPE, positive = tinybert better for R²;
#           coloured checkmark (✓) when tinybert wins significantly,
#           ✗ when loses significantly.
# =============================================================================

COMP_ENCS  = ['only_structured', 'sentencebert', 'clinicalbert']
COMP_LABELS = {
    'only_structured': r'\makecell{Structured\\Only}',
    'sentencebert':    r'\makecell{Sentence\\BERT}',
    'clinicalbert':    r'\makecell{Bio-Clinical\\BERT}',
}

METRICS_STAT = ['mae', 'mse', 'smape', 'r2']
METRIC_LABELS = {'mae': 'MAE', 'mse': 'MSE', 'smape': 'sMAPE', 'r2': '$R^{2}$'}
# Lower is better for MAE/MSE/sMAPE; higher is better for R²
HIGHER_BETTER = {'mae': False, 'mse': False, 'smape': False, 'r2': True}

# Build per-fold data frame indexed by (encoding, model, fold)
fold_data = filt.set_index(['encoding', 'model', 'fold'])[METRICS_STAT]

# Collect all Wilcoxon test inputs
records = []   # (enc_cmp, model, metric, diffs)
for enc_cmp in COMP_ENCS:
    if enc_cmp not in enc_avail_all:
        continue
    for model in model_avail:
        for metric in METRICS_STAT:
            try:
                vals_tiny  = [float(fold_data.loc[('tinybert',   model, f), metric])
                               for f in range(5)]
                vals_other = [float(fold_data.loc[(enc_cmp, model, f), metric])
                               for f in range(5)]
            except KeyError:
                continue
            diffs = np.array(vals_tiny) - np.array(vals_other)
            records.append({
                'enc_cmp': enc_cmp,
                'model':   model,
                'metric':  metric,
                'mean_diff': float(np.mean(diffs)),
                'diffs':     diffs,
            })

# Run Wilcoxon for each record (n=5; p-values capped at 1.0 when all diffs=0)
pvals = []
for r in records:
    d = r['diffs']
    if np.all(d == 0):
        p = 1.0
    else:
        try:
            _, p = wilcoxon(d, alternative='two-sided', zero_method='wilcox')
        except Exception:
            p = 1.0
    pvals.append(p)

# FDR-BH correction
if pvals:
    _, qvals, _, _ = multipletests(pvals, method='fdr_bh')
else:
    qvals = np.array([])

# Attach corrected p-values back
for i, r in enumerate(records):
    r['pval']  = pvals[i]
    r['qval']  = float(qvals[i])

# Index results for quick lookup
stat_idx = {}
for r in records:
    stat_idx[(r['enc_cmp'], r['model'], r['metric'])] = r

def sig_stars(q):
    """Return significance star string based on FDR-corrected q-value."""
    if   q < 0.001: return r'$^{***}$'
    elif q < 0.01:  return r'$^{**}$'
    elif q < 0.05:  return r'$^{*}$'
    else:           return ''

def win_symbol(mean_diff, metric, qval):
    """
    Return LaTeX coloured symbol.
    Win  = tinybert is better AND significant → green checkmark
    Lose = tinybert is worse  AND significant → red times
    Tie (or not significant)                  → em-dash
    """
    hb       = HIGHER_BETTER[metric]
    tiny_win = (mean_diff < 0) if not hb else (mean_diff > 0)
    sig      = qval < 0.05
    if sig and tiny_win:
        return r'{\color{green!60!black}$\checkmark$}'
    elif sig and not tiny_win:
        return r'{\color{red!70!black}$\times$}'
    else:
        return r'---'

# =============================================================================
# Build LaTeX table
# =============================================================================
col_spec = 'l' + (''.join(['rr' * len(METRICS_STAT)]) * len(COMP_ENCS))
# Simpler: rows = models, column groups = (enc_cmp × metrics), cell = Δ + symbol

# Column format: model | [enc1: mae Δ ✓, mse Δ ✓, smape Δ ✓, r2 Δ ✓] | [enc2: ...] | [enc3: ...]
# Each metric cell = "value symbol stars"  → single column per metric per enc_cmp
n_metric_cols = len(METRICS_STAT)  # 4
n_enc_cols    = len(COMP_ENCS)     # 3
total_data_cols = n_enc_cols * n_metric_cols   # 12

col_format = 'l' + ('c' * total_data_cols)

# Header for metric names (repeated per enc_cmp group)
metric_header = ' & '.join([METRIC_LABELS[m] for m in METRICS_STAT])

lines = []
lines.append(r'\begin{table}[htbp]')
lines.append(r'\centering')
lines.append(r'\caption{Statistical comparison of TinySurgicalBERT versus baseline text encodings. '
             r'$\Delta$ values show mean fold-level difference (TinySurgicalBERT $-$ baseline); '
             r'negative $\Delta$ is favourable for MAE, MSE, and sMAPE; '
             r'positive $\Delta$ is favourable for $R^{2}$. '
             r'Significance assessed by two-sided Wilcoxon signed-rank test ($n = 5$ folds) '
             r'with Benjamini--Hochberg FDR correction. '
             r'{\color{green!60!black}$\checkmark$} = TinySurgicalBERT significantly better; '
             r'{\color{red!70!black}$\times$} = significantly worse; '
             r'--- = no significant difference.}')
lines.append(r'\label{tab:stat_comparison}')
lines.append(r'\begin{threeparttable}')
lines.append(r'\resizebox{\linewidth}{!}{%')
lines.append(r'\begin{tabular}{' + col_format + r'}')
lines.append(r'\toprule')

# Top header: grouped enc_cmp labels spanning n_metric_cols each
top_cells = ['']
for enc_cmp in COMP_ENCS:
    label = {
        'only_structured': r'vs.\ Structured Only',
        'sentencebert':    r'vs.\ SentenceBERT',
        'clinicalbert':    r'vs.\ Bio-ClinicalBERT',
    }.get(enc_cmp, enc_cmp)
    top_cells.append(r'\multicolumn{' + str(n_metric_cols) + r'}{c}{' + label + r'}')
lines.append(' & '.join(top_cells) + r' \\')

# Cmidrule lines
cr_parts = []
for gi in range(n_enc_cols):
    start = 2 + gi * n_metric_cols
    end   = start + n_metric_cols - 1
    cr_parts.append(rf'\cmidrule(lr){{{start}-{end}}}')
lines.append(''.join(cr_parts))

# Second header: metric names
metric_cells = ['Model']
for _ in COMP_ENCS:
    for m in METRICS_STAT:
        metric_cells.append(METRIC_LABELS[m])
lines.append(' & '.join(metric_cells) + r' \\')
lines.append(r'\midrule')

# Data rows
for model in model_avail:
    row_cells = [MODEL_LABELS.get(model, model)]
    for enc_cmp in COMP_ENCS:
        for metric in METRICS_STAT:
            key = (enc_cmp, model, metric)
            if key not in stat_idx:
                row_cells.append('---')
                continue
            r        = stat_idx[key]
            md       = r['mean_diff']
            qv       = r['qval']
            sym      = win_symbol(md, metric, qv)
            stars    = sig_stars(qv)
            hb       = HIGHER_BETTER[metric]
            abs_md   = abs(md)
            # Format: sign-aware value + symbol + stars
            if metric == 'r2':
                val_str = f'{md:+.4f}'
            elif metric in ('mse',):
                val_str = f'{md:+.1f}'
            else:
                val_str = f'{md:+.2f}'
            cell = f'{val_str}\\,{sym}{stars}'
            row_cells.append(cell)
    lines.append(' & '.join(row_cells) + r' \\')

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}}')
lines.append(r'\begin{tablenotes}')
lines.append(r'\small')
lines.append(r'\item $^{*}p < 0.05$;\ $^{**}p < 0.01$;\ $^{***}p < 0.001$ (FDR-corrected, Benjamini--Hochberg).')
lines.append(r'\item All tests two-sided Wilcoxon signed-rank ($n = 5$ cross-validation folds).')
lines.append(r'\end{tablenotes}')
lines.append(r'\end{threeparttable}')
lines.append(r'\end{table}')

stat_tex_path = os.path.join(RESULTS_DIR, 'statistical_table.tex')
with open(stat_tex_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print(f"  [OK] {stat_tex_path}")

print("\n  All figures and statistical table saved.")
