# TinySurgicalBERT — Section 3: Text Encoding and BERT

## Introduction

The most powerful feature in the surgical dataset is a string of text: the procedure name.
To use this string in any regression model we must convert it into a fixed-length vector of
numbers — a process called **text encoding** or **embedding**.
This section builds up from the simplest possible text encoding through the full
Transformer self-attention mechanism that makes BERT work, explaining every mathematical
piece along the way.

---

## 3.1 From Words to Numbers: The Embedding Concept

### What It Is

A word embedding is a function that maps a word (or token) to a point in a
high-dimensional vector space such that **semantically similar words end up geometrically close**.

**Real-world analogy**: think of a city map where every restaurant is plotted as a dot.
Italian restaurants cluster in one neighbourhood, sushi bars cluster elsewhere, and
"fusion" places sit in between.
Word embeddings do the same thing for words — "appendectomy," "laparotomy," and "cholecystectomy"
cluster close together in embedding space because they all denote abdominal procedures.
"Arthroplasty" sits in a distant cluster because it refers to joint replacement.
A model that can measure these distances can use them to predict surgical duration.

### Why This Exists

A raw word like "appendectomy" has no mathematical meaning to a neural network.
If we assign it the integer 1427 (its index in a vocabulary list) and
"cholecystectomy" the integer 892, the model treats 1427 as numerically 535 units
away from 892 — which is meaningless.
Embeddings replace these arbitrary integers with dense vectors that encode genuine
semantic relationships.

---

## 3.2 BPE Tokenisation

### What It Is

Before a model can embed a word, it must split text into **tokens** — the atomic units
the model processes.
Byte Pair Encoding (BPE) is the tokenisation algorithm used by most modern language models
including TinySurgicalBERT.

**Real-world analogy**: think of how a child learns to read.
First they learn the alphabet (individual characters), then common syllables ("tion", "ing"),
then full words. BPE does the same thing algorithmically — it starts with individual characters
and repeatedly merges the most frequent adjacent pair into a single token until a target
vocabulary size is reached.

### The BPE Algorithm

```{.graphviz}
digraph BPE {
    graph [fontsize=20, dpi=150, size="9,6", ratio=auto,
           margin=0.2, nodesep=0.6, ranksep=0.5,
           fontname="DejaVu Sans", bgcolor="transparent"];
    node  [shape=box, style="rounded,filled", fontsize=17,
           fontname="DejaVu Sans", fontcolor=white, margin=0.18];
    edge  [fontsize=16, penwidth=2, arrowsize=1.2,
           color="#F57C00", fontname="DejaVu Sans"];
    rankdir=LR;

    subgraph cluster_init {
        style=filled; fillcolor="#0D3B6E"; fontcolor=white;
        fontname="DejaVu Sans"; fontsize=17; label="Step 1 — Initial vocab";
        A [label="Vocabulary:\nAll unique characters", fillcolor="#1976D2"];
    }
    subgraph cluster_count {
        style=filled; fillcolor="#1B3A1B"; fontcolor=white;
        fontname="DejaVu Sans"; fontsize=17; label="Step 2 — Count pairs";
        B [label="Find most frequent\nadjacent pair in corpus", fillcolor="#388E3C"];
    }
    subgraph cluster_merge {
        style=filled; fillcolor="#3E0A6E"; fontcolor=white;
        fontname="DejaVu Sans"; fontsize=17; label="Step 3 — Merge";
        C [label="Add merged pair\nto vocabulary", fillcolor="#7B1FA2"];
    }
    subgraph cluster_stop {
        style=filled; fillcolor="#5C1A00"; fontcolor=white;
        fontname="DejaVu Sans"; fontsize=17; label="Stop condition";
        D [label="Vocabulary size\nreaches target\n(e.g. 2,500 tokens)", fillcolor="#BF360C"];
    }

    A -> B -> C -> B [label="repeat"];
    C -> D [label="when done"];
}
```

**Why domain-specific BPE matters for surgical text:**
A standard BPE tokeniser trained on Wikipedia has no entry for "cholecystectomy."
It fragments it as: `cho`, `le`, `cys`, `tec`, `tomy` — 5 tokens for one concept.
A tokeniser trained on our 180,370 surgical procedure descriptions learns that
"cholecystectomy" appears 4,200 times and keeps it as a single token.
Our TinySurgicalBERT corpus achieves 0% unknown-token rate (every procedure name
is handled without falling back to character-level fragmentation).

### Mathematical Formulation

Let $\mathcal{V}$ be the current vocabulary and $\mathcal{C}$ be the corpus of tokenised texts.
Define the **pair frequency**:

$$\text{freq}(a, b) = \sum_{\text{word } w \in \mathcal{C}} \text{count}(a b \text{ occurs adjacent in } w)$$

The merge step selects:

$$(\hat{a}, \hat{b}) = \underset{(a,b)}{\arg\max}\; \text{freq}(a, b)$$

and adds the merged token $\hat{a}\hat{b}$ to $\mathcal{V}$.

**Symbol definitions:**

- $a, b$: adjacent tokens in the current vocabulary
- $\text{freq}(a, b)$: number of times token $a$ is immediately followed by token $b$ across all words in the corpus
- $(\hat{a}, \hat{b})$: the highest-frequency adjacent pair — the next merge operation
- $\mathcal{V}$: the current vocabulary (grows by one entry per iteration)

### Numerical Example

Corpus (simplified): two procedure names appearing many times:
- "lap chole" (laparoscopic cholecystectomy): 4200 times
- "open chole" (open cholecystectomy): 800 times

Initial character-level vocabulary: `{l, a, p, c, h, o, e, n, ...}`

Iteration 1: most frequent adjacent pair across all examples:
- `c` followed by `h` appears $5000$ times (in "chole" both procedures)
- Merge → new token `ch`, add to vocabulary

Iteration 2:
- `ch` followed by `o` appears $5000$ times
- Merge → new token `cho`

After enough iterations, `cholecystectomy` becomes a single token because it appears
5000 times and all its subparts are learned progressively.

**Interpretation**: A domain-trained tokeniser produces shorter, more meaningful token sequences
for surgical text.
Fewer tokens means the BERT model processes each procedure with fewer attention operations,
and each token carries richer semantic meaning.

---

## 3.3 The Transformer Self-Attention Mechanism

### What It Is

Self-attention is the core computation inside BERT.
It allows every token in a sequence to look at every other token and decide how much
"attention" to pay to each one when computing its own updated representation.

**Real-world analogy**: imagine you are reading the procedure "bilateral total knee arthroplasty."
When you process "knee," you naturally look back at "bilateral" to know this is a two-sided
procedure and forward to "arthroplasty" to know the surgery type.
Self-attention automates this cross-reference for every word simultaneously.

### Mathematical Foundation

**Intuition (before symbols)**: Self-attention takes a sequence of token vectors, and for each
token, computes a weighted average of all other token vectors.
The weights are determined by how "relevant" each token is to the current token —
computed via a dot-product similarity between learned projections.
Tokens that are highly relevant get high weight; unrelated tokens get near-zero weight.

**Setup**: We have $L$ tokens, each represented as a $d_{\text{model}}$-dimensional vector.
We stack these into a matrix $\mathbf{X} \in \mathbb{R}^{L \times d_{\text{model}}}$.

**Three learned projection matrices:**

$$\mathbf{Q} = \mathbf{X}\mathbf{W}^Q, \qquad \mathbf{K} = \mathbf{X}\mathbf{W}^K, \qquad \mathbf{V} = \mathbf{X}\mathbf{W}^V$$

**The attention output:**

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

**Symbol definitions:**

- $\mathbf{X} \in \mathbb{R}^{L \times d_{\text{model}}}$: input matrix; row $i$ is the embedding of token $i$
- $L$: sequence length (number of tokens)
- $d_{\text{model}}$: embedding dimensionality (128 in TinySurgicalBERT, 768 in BERT-base)
- $\mathbf{W}^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$: learned "Query" projection weight matrix
- $\mathbf{W}^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$: learned "Key" projection weight matrix
- $\mathbf{W}^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$: learned "Value" projection weight matrix
- $\mathbf{Q} \in \mathbb{R}^{L \times d_k}$: Query matrix — what each token is "looking for"
- $\mathbf{K} \in \mathbb{R}^{L \times d_k}$: Key matrix — what each token "advertises"
- $\mathbf{V} \in \mathbb{R}^{L \times d_v}$: Value matrix — what each token "contributes" if selected
- $d_k$: query/key dimensionality (head size, often $d_{\text{model}} / h$ where $h$ = number of heads)
- $\mathbf{Q}\mathbf{K}^\top \in \mathbb{R}^{L \times L}$: raw attention score matrix — entry $(i,j)$ measures how relevant token $j$ is to token $i$
- $\sqrt{d_k}$: scaling factor preventing dot products from growing too large in magnitude (which would saturate the softmax gradients)
- $\text{softmax}(\cdot)$: applied row-wise, converts raw scores to probabilities in $[0,1]$ summing to 1
- $\text{softmax}(\cdot) \mathbf{V}$: weighted average of Value vectors, where weights are attention probabilities

**Term-by-term breakdown:**

1. **$\mathbf{Q}\mathbf{K}^\top$**: The dot product between row $i$ of $\mathbf{Q}$ and row $j$ of $\mathbf{K}$ measures similarity between what token $i$ is looking for and what token $j$ offers. A high dot product means "token $j$ is very relevant to token $i$'s context."

2. **Division by $\sqrt{d_k}$**: Without this, dot products in high-dimensional spaces become very large (variance scales with $d_k$), pushing softmax into extreme regions where gradients vanish. Dividing by $\sqrt{d_k}$ keeps the variance at 1 regardless of dimensionality.

3. **$\text{softmax}(\cdot)$**: Converts raw scores for each row into attention weights that sum to 1. Token $i$ now has a probability distribution over all $L$ tokens indicating how much to "attend to" each.

4. **$\times \mathbf{V}$**: The final output for token $i$ is a weighted average of all Value vectors, with weights from step 3. If token $i$ attends heavily to token $j$, then token $j$'s Value vector dominates token $i$'s output — effectively "borrowing" information from it.

### Numerical Example

Toy sequence: 3 tokens (L=3), $d_k = 2$ (just 2 dimensions for clarity).

Query matrix $\mathbf{Q}$ (each row = one token's query vector):

$$\mathbf{Q} = \begin{bmatrix} 1.0 & 0.5 \\ 0.2 & 1.0 \\ 0.8 & 0.3 \end{bmatrix}$$

Key matrix $\mathbf{K}$:

$$\mathbf{K} = \begin{bmatrix} 1.0 & 0.3 \\ 0.5 & 0.9 \\ 0.1 & 0.7 \end{bmatrix}$$

**Step 1**: Compute raw scores $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top$:

$$S_{11} = 1.0 \times 1.0 + 0.5 \times 0.3 = 1.15$$
$$S_{12} = 1.0 \times 0.5 + 0.5 \times 0.9 = 0.95$$
$$S_{13} = 1.0 \times 0.1 + 0.5 \times 0.7 = 0.45$$

Full matrix for row 1 (token 1's scores toward tokens 1, 2, 3):

$$\mathbf{S}_{1,:} = [1.15,\; 0.95,\; 0.45]$$

**Step 2**: Scale by $\sqrt{d_k} = \sqrt{2} \approx 1.414$:

$$\frac{\mathbf{S}_{1,:}}{\sqrt{2}} = [0.813,\; 0.672,\; 0.318]$$

**Step 3**: Softmax over scaled scores:

$$e^{0.813} = 2.255, \quad e^{0.672} = 1.958, \quad e^{0.318} = 1.374$$

$$\text{sum} = 2.255 + 1.958 + 1.374 = 5.587$$

$$\mathbf{A}_{1,:} = \left[\frac{2.255}{5.587},\; \frac{1.958}{5.587},\; \frac{1.374}{5.587}\right] = [0.404,\; 0.350,\; 0.246]$$

**Interpretation**: Token 1 attends to itself (40.4%), to token 2 (35.0%), and to token 3 (24.6%).
In the procedure "bilateral total knee arthroplasty," if token 1 is "knee," it pays more
attention to "total" (token 2) than to "bilateral" (token 3) — which makes clinical sense,
since "total" directly qualifies the surgery type.

```{.matplotlib}
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor('#0A0A0A')

# Left: attention matrix heatmap (toy 3-token example)
attn = np.array([[0.404, 0.350, 0.246],
                 [0.150, 0.560, 0.290],
                 [0.220, 0.310, 0.470]])

for ax in (ax1, ax2):
    ax.set_facecolor('#0A0A0A')
    for spine in ax.spines.values():
        spine.set_color('#444444')
    ax.tick_params(colors='#CCCCCC', labelsize=11)

im = ax1.imshow(attn, cmap='Blues', vmin=0, vmax=0.6)
plt.colorbar(im, ax=ax1)
tokens = ['knee', 'total', 'bilateral']
ax1.set_xticks([0,1,2]); ax1.set_xticklabels(tokens, color='#CCCCCC', fontsize=11)
ax1.set_yticks([0,1,2]); ax1.set_yticklabels(tokens, color='#CCCCCC', fontsize=11)
ax1.set_xlabel('Attends TO', color='#CCCCCC')
ax1.set_ylabel('FROM token', color='#CCCCCC')
ax1.set_title('Self-attention weights (toy)', color='#CCCCCC', fontsize=12)
for i in range(3):
    for j in range(3):
        ax1.text(j, i, f'{attn[i,j]:.2f}', ha='center', va='center',
                 color='white' if attn[i,j] > 0.35 else '#CCCCCC', fontsize=12)

# Right: how embeddings are updated
heads = ['h=1\nSyntax', 'h=2\nProcedure', 'h=3\nLaterality', 'h=4\nModifier']
scores = [0.71, 0.89, 0.64, 0.77]
colors = ['#1565C0', '#2E7D32', '#C62828', '#6A1B9A']
bars = ax2.bar(heads, scores, color=colors, width=0.55, edgecolor='none')
ax2.set_ylim(0, 1.1)
ax2.set_ylabel('Average attention weight', color='#CCCCCC')
ax2.set_title('Attention head specialisation\n(representative)', color='#CCCCCC', fontsize=12)
for bar, s in zip(bars, scores):
    ax2.text(bar.get_x() + bar.get_width()/2, s + 0.03, f'{s:.2f}',
             ha='center', color='#CCCCCC', fontsize=11)
ax2.xaxis.label.set_color('#CCCCCC')
ax2.yaxis.label.set_color('#CCCCCC')

plt.tight_layout()
```

**What to observe:**

- **Left heatmap**: Each row shows which tokens a given token attends to.
  "total" (row 2) attends strongly to itself (0.56) — this token is highly self-informative.
  "knee" (row 1) distributes attention across all tokens, collecting context from both modifiers.
- **Right bar chart**: In a multi-head model, different attention heads specialise in different
  aspects of the procedure name — syntax, procedure type, laterality, and modifiers.
  The model learns this specialisation automatically from data.

---

## 3.4 BERT: Bidirectional Context

### What It Is

BERT (Bidirectional Encoder Representations from Transformers) stacks multiple self-attention
layers and processes each token in the **context of all other tokens simultaneously** —
both the words that come before and after it.
Earlier language models were unidirectional (left-to-right only); BERT broke this limitation.

```{.graphviz}
digraph BERT {
    graph [fontsize=20, dpi=150, size="9,7", ratio=auto,
           margin=0.2, nodesep=0.5, ranksep=0.5,
           fontname="DejaVu Sans", bgcolor="transparent"];
    node  [shape=box, style="rounded,filled", fontsize=17,
           fontname="DejaVu Sans", fontcolor=white, margin=0.18];
    edge  [fontsize=15, penwidth=2, arrowsize=1.2,
           color="#F57C00", fontname="DejaVu Sans"];

    rankdir=TB;

    subgraph cluster_input {
        style=filled; fillcolor="#0D3B6E"; fontcolor=white;
        fontname="DejaVu Sans"; fontsize=17; label="Input Layer";
        t1 [label="[CLS]", fillcolor="#1976D2"];
        t2 [label="bilateral", fillcolor="#1976D2"];
        t3 [label="knee", fillcolor="#1976D2"];
        t4 [label="arthroplasty", fillcolor="#1976D2"];
    }
    subgraph cluster_l1 {
        style=filled; fillcolor="#1B3A1B"; fontcolor=white;
        fontname="DejaVu Sans"; fontsize=17; label="Transformer Layer 1 (Self-Attention + FFN)";
        h1 [label="h₁", fillcolor="#388E3C"];
        h2 [label="h₂", fillcolor="#388E3C"];
        h3 [label="h₃", fillcolor="#388E3C"];
        h4 [label="h₄", fillcolor="#388E3C"];
    }
    subgraph cluster_l2 {
        style=filled; fillcolor="#3E0A6E"; fontcolor=white;
        fontname="DejaVu Sans"; fontsize=17; label="Transformer Layer 2 (Self-Attention + FFN)";
        h5 [label="h₁'", fillcolor="#7B1FA2"];
        h6 [label="h₂'", fillcolor="#7B1FA2"];
        h7 [label="h₃'", fillcolor="#7B1FA2"];
        h8 [label="h₄'", fillcolor="#7B1FA2"];
    }
    subgraph cluster_pool {
        style=filled; fillcolor="#5C1A00"; fontcolor=white;
        fontname="DejaVu Sans"; fontsize=17; label="Pooling";
        cls_out [label="[CLS] vector\n= sentence embedding", fillcolor="#BF360C"];
    }

    t1 -> h1; t2 -> h2; t3 -> h3; t4 -> h4;
    h1 -> h5; h2 -> h6; h3 -> h7; h4 -> h8;
    h1 -> h6; h1 -> h7; h1 -> h8;   // bidirectional connections
    h4 -> h5; h4 -> h6; h4 -> h7;   // every token attends to every token
    h5 -> cls_out;
}
```

**What to observe**: Every token at Layer 1 feeds into every token at Layer 2 (the crossing edges).
This is bidirectionality — "arthroplasty" influences the representation of "bilateral"
just as much as "bilateral" influences "arthroplasty."
The final `[CLS]` vector at the top is extracted as the sentence embedding representing the
entire procedure name.

### CLS Pooling

BERT inserts a special `[CLS]` token at the beginning of every input.
After passing through all layers, this token's final hidden state is used as the
**sentence-level embedding** representing the entire procedure description.

$$\mathbf{e}_i^{\text{text}} = \mathbf{H}_{i,\,[CLS]}^{(L_{\text{model}})}$$

- $\mathbf{H}^{(L_{\text{model}})}$: the hidden state matrix after the final transformer layer
- $[CLS]$: the index of the special classification token (always position 0)
- $\mathbf{e}_i^{\text{text}} \in \mathbb{R}^{d_{\text{model}}}$: the embedding for procedure $i$

For TinySurgicalBERT: $d_{\text{model}} = 128$, so each procedure name becomes a 128-dimensional vector.

---

## 3.5 The Four Encoding Strategies Compared

| Encoding | Model | Trained on | Output dim | Size | 0% UNK? |
|---|---|---|---|---|---|
| Structured Only | None | — | 0 (no text) | 0 MB | — |
| SentenceBERT | SBERT | General English | 384 | ~90 MB | No |
| Bio-ClinicalBERT | BERT-base | Clinical notes | 768 | 440 MB | No |
| TinySurgicalBERT | Custom | Our 180K procedures | 128 | 0.75 MB | ✅ Yes |

**Key insight**: TinySurgicalBERT is 614× smaller than Bio-ClinicalBERT because:

1. It uses only 2 transformer layers instead of 12
2. Its hidden dimension is 128 instead of 768
3. Its vocabulary is 2,500 surgical-specific tokens instead of 30,000 general tokens
4. It is INT8-quantised (weights stored as 8-bit integers instead of 32-bit floats)

Despite being 614× smaller, it achieves essentially identical downstream predictive accuracy
(MAE difference: 0.02 minutes = 1.2 seconds).

---

## 3.6 Code: Running Each Text Encoder

```python
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import onnxruntime as ort
import numpy as np

# Example procedure names
procedures = [
    "bilateral total knee arthroplasty with cemented fixation",
    "laparoscopic cholecystectomy",
    "open Whipple procedure with portal vein resection",
]

# --- SentenceBERT (384-d) ---
sbert = SentenceTransformer('all-MiniLM-L6-v2')
emb_sbert = sbert.encode(procedures)   # shape: (3, 384)
print(f"SentenceBERT shape: {emb_sbert.shape}")
# Output: SentenceBERT shape: (3, 384)

# --- Bio-ClinicalBERT (768-d) ---
tok = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
mdl = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
mdl.eval()
with torch.no_grad():
    enc = tok(procedures, padding=True, truncation=True, return_tensors='pt', max_length=64)
    out = mdl(**enc)
emb_clin = out.last_hidden_state[:, 0, :].numpy()  # CLS token
print(f"Bio-ClinicalBERT shape: {emb_clin.shape}")
# Output: Bio-ClinicalBERT shape: (3, 768)

# --- TinySurgicalBERT (ONNX INT8, 128-d) ---
session = ort.InferenceSession('./models/tinybert_int8.onnx')
# (tokenisation using the trained domain BPE tokeniser happens in pipeline.py)
# emb_tiny = session.run(None, {'input_ids': ..., 'attention_mask': ...})
# Output shape: (3, 128)  -- runs in 0.64 ms per case on CPU
```

---

## Summary

| Concept | Key Takeaway |
|---|---|
| Word embedding | Maps tokens to vectors where similar tokens are geometrically close |
| BPE tokenisation | Merges frequent adjacent pairs; domain-specific BPE gets 0% UNK on surgical text |
| Self-attention | Each token attends to all others; weights determined by Query-Key dot products |
| Scaling by $\sqrt{d_k}$ | Prevents dot products from growing too large and saturating softmax |
| CLS pooling | Final `[CLS]` hidden state summarises the entire procedure name as one vector |
| TinySurgicalBERT | 614× smaller than teacher while matching its predictive accuracy |
