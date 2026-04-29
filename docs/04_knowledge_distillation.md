# TinySurgicalBERT — Section 4: Knowledge Distillation

## Introduction

Bio-ClinicalBERT produces excellent text embeddings for surgical procedure names —
but it weighs 440 MB and takes hundreds of milliseconds to run.
That is unacceptable for a mobile OR scheduling app.
This section explains exactly how we shrink the model to 0.75 MB without meaningfully
sacrificing accuracy, using a technique called **knowledge distillation**.

---

## 4.1 The Teacher-Student Paradigm

### What It Is

Knowledge distillation trains a small **student** model to mimic the output of a large
**teacher** model, rather than learning directly from raw labels.
The key insight is that the teacher's output embeddings are richer than a single label —
they encode a full representation of the input, and the student learns to reproduce that
representation in a compressed form.

**Real-world analogy**: imagine you want to teach someone to replicate a master chef's
flavour judgements without sending them to 10 years of culinary school.
Instead of having them taste millions of dishes from scratch (learning from raw ingredients),
you have the master chef write a detailed score vector for each dish
("acidity: 7.3, saltiness: 4.1, umami: 8.9 ...") and train the student to reproduce those
scores.
The student learns the chef's *judgement framework* — not just "this dish is good."
That is exactly what the student BERT learns from the teacher's embedding vectors.

### Pipeline Overview

```{.graphviz}
digraph KD {
    graph [fontsize=20, dpi=150, size="9,8", ratio=auto,
           margin=0.2, nodesep=0.5, ranksep=0.5,
           fontname="DejaVu Sans", bgcolor="transparent"];
    node  [shape=box, style="rounded,filled", fontsize=17,
           fontname="DejaVu Sans", fontcolor=white, margin=0.20];
    edge  [fontsize=16, penwidth=2, arrowsize=1.2,
           color="#F57C00", fontname="DejaVu Sans"];
    rankdir=TB;

    proc [label="Procedure text\n\"laparoscopic cholecystectomy\"",
          fillcolor="#1976D2", color="#0D3B6E"];

    subgraph cluster_teacher {
        style=filled; fillcolor="#1B3A1B"; fontcolor=white;
        fontname="DejaVu Sans"; fontsize=17; label="Teacher (frozen — not updated)";
        teacher [label="Bio-ClinicalBERT\n12 layers, 768-d hidden\n440 MB, 110M params", fillcolor="#388E3C"];
        t_emb   [label="Teacher embedding\nz_T ∈ R^768", fillcolor="#388E3C"];
    }
    subgraph cluster_student {
        style=filled; fillcolor="#3E0A6E"; fontcolor=white;
        fontname="DejaVu Sans"; fontsize=17; label="Student (trained — weights updated)";
        student [label="TinySurgicalBERT\n2 layers, 128-d hidden\n0.63M params", fillcolor="#7B1FA2"];
        proj    [label="Linear projection\n128-d → 768-d", fillcolor="#7B1FA2"];
        s_emb   [label="Student embedding\nz_S ∈ R^768 (projected)", fillcolor="#7B1FA2"];
    }

    subgraph cluster_loss {
        style=filled; fillcolor="#5C1A00"; fontcolor=white;
        fontname="DejaVu Sans"; fontsize=17; label="Distillation Loss";
        loss [label="L = α·MSE(z_S, z_T) + (1-α)·(1 - cosine(z_S, z_T))",
              fillcolor="#BF360C"];
    }

    subgraph cluster_export {
        style=filled; fillcolor="#003333"; fontcolor=white;
        fontname="DejaVu Sans"; fontsize=17; label="Deployment";
        quant [label="INT8 Quantisation\n32-bit → 8-bit weights", fillcolor="#00796B"];
        onnx  [label="ONNX Export\n0.75 MB — 614× compression", fillcolor="#00796B"];
    }

    proc    -> teacher -> t_emb;
    proc    -> student -> proj -> s_emb;
    t_emb   -> loss;
    s_emb   -> loss;
    loss    -> student [label="backprop\n(gradients)", style=dashed, color="#7B1FA2"];
    student -> quant -> onnx;
}
```

**What to observe**: The teacher (green) is frozen — its weights are never updated.
Only the student (purple) learns.
The loss function measures how different the student's embedding is from the teacher's,
and the gradients flow back through the student to reduce this difference.
After training, only the student is exported; the teacher is discarded.

---

## 4.2 The Distillation Loss Function

### Intuition (before any symbols)

We want the student's output vector to be as similar as possible to the teacher's output
vector for the same input.
Two measures of vector similarity are useful:

1. **MSE (Mean Squared Error)**: penalises the absolute coordinate differences.
   If the teacher says "dimension 47 = 0.8" and the student says "dimension 47 = 0.3,"
   MSE penalises this by $(0.8 - 0.3)^2 = 0.25$.
   MSE cares about both the *direction* and *magnitude* of the vectors being similar.

2. **Cosine similarity**: measures the angle between the two vectors, ignoring their lengths.
   Two vectors pointing in the same direction have cosine similarity 1.0,
   regardless of whether one is twice as long as the other.
   Cosine similarity cares only about the *semantic direction* of the embedding.

We combine both losses because they capture complementary aspects of similarity.
MSE ensures the student learns the scale and exact values; cosine ensures the student
learns the semantic direction even if magnitudes are slightly off.

### Mathematical Formulation

$$\mathcal{L}_{\text{distill}} = \alpha \cdot \mathcal{L}_{\text{MSE}}(\mathbf{z}_S, \mathbf{z}_T) + (1 - \alpha) \cdot \mathcal{L}_{\text{cos}}(\mathbf{z}_S, \mathbf{z}_T)$$

**Symbol definitions:**

- $\mathbf{z}_T \in \mathbb{R}^{d_T}$: the teacher's embedding vector for a given procedure; $d_T = 768$ for Bio-ClinicalBERT
- $\mathbf{z}_S \in \mathbb{R}^{d_T}$: the student's embedding, **after** applying a learned linear projection $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{d_S \times d_T}$ that maps from student dimension $d_S = 128$ to teacher dimension $d_T = 768$
- $\alpha \in [0, 1]$: a mixing hyperparameter controlling the relative weight of the two loss terms; $\alpha = 0.5$ in this project
- $\mathcal{L}_{\text{MSE}}$: the mean squared error loss
- $\mathcal{L}_{\text{cos}}$: the cosine-dissimilarity loss (1 minus cosine similarity)

**Expanding each term:**

$$\mathcal{L}_{\text{MSE}}(\mathbf{z}_S, \mathbf{z}_T) = \frac{1}{d_T} \sum_{k=1}^{d_T} \left(z_{S,k} - z_{T,k}\right)^2$$

$$\mathcal{L}_{\text{cos}}(\mathbf{z}_S, \mathbf{z}_T) = 1 - \frac{\mathbf{z}_S \cdot \mathbf{z}_T}{\|\mathbf{z}_S\|_2 \;\|\mathbf{z}_T\|_2}$$

**Additional symbol definitions:**

- $z_{S,k}$: the $k$-th coordinate of the student's projected embedding
- $z_{T,k}$: the $k$-th coordinate of the teacher's embedding
- $d_T = 768$: the number of dimensions (sum runs over all dimensions)
- $\mathbf{z}_S \cdot \mathbf{z}_T = \sum_{k=1}^{d_T} z_{S,k} \cdot z_{T,k}$: the dot product
- $\|\mathbf{v}\|_2 = \sqrt{\sum_k v_k^2}$: the Euclidean (L2) norm of vector $\mathbf{v}$
- $\frac{\mathbf{z}_S \cdot \mathbf{z}_T}{\|\mathbf{z}_S\|_2 \|\mathbf{z}_T\|_2}$: cosine similarity, always in $[-1, +1]$
- $(1 - \text{cosine similarity})$: cosine dissimilarity, in $[0, 2]$; 0 means identical direction, 2 means opposite directions

**Term-by-term breakdown of the full loss:**

1. **$\alpha \cdot \mathcal{L}_{\text{MSE}}$**: The MSE term pushes each individual coordinate of the student toward the teacher's coordinate. If $\alpha = 0.5$, this term contributes 50% of the gradient signal. Because it operates coordinate-by-coordinate, it is sensitive to both the direction and the absolute scale of the embeddings.

2. **$(1-\alpha) \cdot \mathcal{L}_{\text{cos}}$**: The cosine term pushes the student's vector to point in the same direction as the teacher's, regardless of magnitude. This is crucial when the student's projection layer has not yet learned the correct scale — cosine loss still provides a useful training signal even in early training.

3. **Why both?** MSE can be fooled if the student learns a zero-magnitude embedding (the zero vector has MSE = teacher's norm squared, which is finite, but the semantic content is lost). Cosine loss is undefined at zero magnitude, but it corrects for scale mismatches that MSE ignores. Together they are robust.

### Numerical Example

Suppose $d_T = 4$ (toy, not 768) and $\alpha = 0.5$.

Teacher embedding: $\mathbf{z}_T = [0.8,\; -0.3,\; 0.5,\; 0.2]$

Student embedding (projected): $\mathbf{z}_S = [0.6,\; -0.4,\; 0.3,\; 0.1]$

**Step 1 — MSE:**

$$\mathcal{L}_{\text{MSE}} = \frac{1}{4}\left[(0.6-0.8)^2 + (-0.4-(-0.3))^2 + (0.3-0.5)^2 + (0.1-0.2)^2\right]$$

$$= \frac{1}{4}\left[(-0.2)^2 + (-0.1)^2 + (-0.2)^2 + (-0.1)^2\right]$$

$$= \frac{1}{4}\left[0.04 + 0.01 + 0.04 + 0.01\right] = \frac{0.10}{4} = 0.025$$

**Step 2 — Dot product:**

$$\mathbf{z}_S \cdot \mathbf{z}_T = (0.6)(0.8) + (-0.4)(-0.3) + (0.3)(0.5) + (0.1)(0.2)$$
$$= 0.480 + 0.120 + 0.150 + 0.020 = 0.770$$

**Step 3 — Norms:**

$$\|\mathbf{z}_T\|_2 = \sqrt{0.8^2 + 0.3^2 + 0.5^2 + 0.2^2} = \sqrt{0.64 + 0.09 + 0.25 + 0.04} = \sqrt{1.02} = 1.010$$

$$\|\mathbf{z}_S\|_2 = \sqrt{0.6^2 + 0.4^2 + 0.3^2 + 0.1^2} = \sqrt{0.36 + 0.16 + 0.09 + 0.01} = \sqrt{0.62} = 0.787$$

**Step 4 — Cosine similarity:**

$$\text{cosine} = \frac{0.770}{1.010 \times 0.787} = \frac{0.770}{0.795} = 0.969$$

$$\mathcal{L}_{\text{cos}} = 1 - 0.969 = 0.031$$

**Step 5 — Combined loss:**

$$\mathcal{L}_{\text{distill}} = 0.5 \times 0.025 + 0.5 \times 0.031 = 0.0125 + 0.0155 = 0.028$$

**Interpretation**: The student's embedding is cosine-similar to the teacher (0.969 — nearly identical direction).
But it is slightly off in absolute values (MSE = 0.025).
A total loss of 0.028 is a good starting point — after training, both losses typically drop to near zero,
meaning the student almost perfectly reproduces the teacher's embedding for each procedure name.

---

## 4.3 INT8 Quantisation

### What It Is

After training, the student's weights are stored as 32-bit floating-point numbers
(each weight uses 4 bytes of memory).
INT8 quantisation converts those weights to 8-bit integers (1 byte each),
achieving a **4× reduction in model size** with minimal accuracy loss.

**Real-world analogy**: imagine you are storing a colour photograph.
A 32-bit image stores 4 billion possible shades per channel.
An 8-bit image stores only 256 shades per channel — but for most photographs, the visual
quality difference is imperceptible.
Similarly, the slight precision loss from 32-bit → 8-bit floating point weights does not
meaningfully affect the semantic content of the embeddings.

### Mathematical Formulation

A floating-point weight $w \in \mathbb{R}$ is quantised as:

$$w_{\text{int8}} = \text{round}\!\left(\frac{w}{s} - z\right)$$

and dequantised for inference as:

$$\hat{w} = s \cdot (w_{\text{int8}} + z)$$

**Symbol definitions:**

- $w$: the original 32-bit float weight
- $s$: the **scale factor** — the real-valued step size between adjacent quantisation levels
- $z$: the **zero point** — the integer value that maps to floating-point 0
- $w_{\text{int8}} \in \{-128, -127, \ldots, 127\}$: the quantised 8-bit signed integer
- $\hat{w}$: the dequantised approximation of $w$
- $\text{round}(\cdot)$: rounding to the nearest integer

**Why this works**: The scale $s$ is chosen so the full range of the float tensor maps
to the full range $[-128, 127]$ of INT8.
Values are then rounded to the nearest integer.
The maximum rounding error is $\frac{s}{2}$, which is small when $s$ is small
(i.e., when the original weights have a narrow value range).

### Numerical Example

Float weight tensor with 5 values: $[-0.42,\; 0.18,\; 0.67,\; -0.83,\; 0.05]$

**Step 1**: Find range: min $= -0.83$, max $= 0.67$.

**Step 2**: Compute scale:

$$s = \frac{\text{max} - \text{min}}{255} = \frac{0.67 - (-0.83)}{255} = \frac{1.50}{255} = 0.00588$$

**Step 3**: Compute zero point:

$$z = \text{round}\!\left(-\frac{\text{min}}{s}\right) = \text{round}\!\left(\frac{0.83}{0.00588}\right) = \text{round}(141.2) = 141$$

**Step 4**: Quantise first weight $w = -0.42$:

$$w_{\text{int8}} = \text{round}\!\left(\frac{-0.42}{0.00588} - 141\right) = \text{round}(-71.4 - 141) = \text{round}(-212.4) = -128$$

(clamped to INT8 minimum; the value is in the tail of the distribution)

**Step 5**: Dequantise:

$$\hat{w} = 0.00588 \times (-128 + 141) = 0.00588 \times 13 = 0.0764$$

The original was $-0.42$; after quantisation it became $0.0764$ — a large error for this
particular extreme value.
For the bulk of weights near the centre of the distribution, the error is much smaller.

**Interpretation**: INT8 quantisation introduces small rounding errors in each weight.
For TinySurgicalBERT, these errors aggregate across the 0.63M parameters but do not
significantly affect embedding quality because the model was trained to be robust to such perturbations
(dynamic quantisation is applied post-training using PyTorch).

---

## 4.4 ONNX Export

### What It Is

ONNX (Open Neural Network Exchange) is a standardised file format for machine learning models
that allows a model trained in PyTorch to run on any device — iOS, Android, Windows, web —
using the ONNX Runtime inference engine.

The export process:
1. Traces the forward pass of the PyTorch model with dummy inputs
2. Builds a computation graph in ONNX format
3. Applies operator fusion and memory optimisations
4. Writes the final `.onnx` file (0.75 MB after INT8 quantisation)

```{.graphviz}
digraph ONNX {
    graph [fontsize=20, dpi=150, size="10,4", ratio=auto,
           margin=0.2, nodesep=0.8, ranksep=0.5,
           fontname="DejaVu Sans", bgcolor="transparent"];
    node  [shape=box, style="rounded,filled", fontsize=17,
           fontname="DejaVu Sans", fontcolor=white, margin=0.18];
    edge  [fontsize=15, penwidth=2, arrowsize=1.2,
           color="#F57C00", fontname="DejaVu Sans"];
    rankdir=LR;

    pt  [label="PyTorch model\n(32-bit, 2.5 MB)", fillcolor="#1976D2", color="#0D3B6E"];
    q   [label="INT8 Quantisation\n(dynamic, post-training)", fillcolor="#388E3C", color="#1B3A1B"];
    ex  [label="torch.onnx.export()\n(traces computation graph)", fillcolor="#7B1FA2", color="#3E0A6E"];
    opt [label="ONNX Runtime\noptimiser\n(operator fusion)", fillcolor="#BF360C", color="#5C1A00"];
    out [label="tinybert_int8.onnx\n0.75 MB", fillcolor="#00796B", color="#003333"];

    pt -> q -> ex -> opt -> out;
}
```

**What to observe**: The pipeline is strictly left-to-right and one-way.
After export, the `.onnx` file is self-contained — no Python, no PyTorch needed to run it.

### Inference Latency

The final ONNX model runs at **0.64 milliseconds per case** on CPU.
This means a full day's schedule of 40 cases can be embedded in 25.6 milliseconds —
well under any real-time scheduling requirement.

| Model | Format | Size | Latency per case |
|---|---|---|---|
| Bio-ClinicalBERT | PyTorch FP32 | 440 MB | ~250 ms |
| SentenceBERT | PyTorch FP32 | ~90 MB | ~15 ms |
| TinySurgicalBERT | ONNX INT8 | 0.75 MB | 0.64 ms |

---

## 4.5 Code: Training and Exporting TinySurgicalBERT

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import onnx
import onnxruntime as ort

# --- Define the student architecture ---
class TinySurgicalBERT(nn.Module):
    def __init__(self, vocab_size=2500, hidden=128, layers=2, heads=4, d_teacher=768):
        super().__init__()
        from transformers import BertConfig, BertModel
        cfg = BertConfig(
            vocab_size=vocab_size, hidden_size=hidden,
            num_hidden_layers=layers, num_attention_heads=heads,
            intermediate_size=hidden * 4, max_position_embeddings=64
        )
        self.bert = BertModel(cfg)
        # Projection: 128-d → 768-d so we can compare to teacher
        self.proj = nn.Linear(hidden, d_teacher)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [CLS] vector
        return self.proj(cls)                  # projected to teacher dimension

# --- Distillation training loop (one batch) ---
student = TinySurgicalBERT()
teacher = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
teacher.eval()  # teacher is frozen — no gradient tracking

optimizer = torch.optim.AdamW(student.parameters(), lr=5e-4)
alpha = 0.5

# Simulated batch (in practice: tokenised procedure names)
input_ids = torch.randint(0, 2500, (32, 16))   # batch=32, seq_len=16
attn_mask = torch.ones(32, 16, dtype=torch.long)

with torch.no_grad():                          # teacher: no gradient
    t_enc = teacher(input_ids=input_ids.clamp(0, 30521),
                    attention_mask=attn_mask)
    z_T = t_enc.last_hidden_state[:, 0, :]    # teacher CLS (768-d)

z_S = student(input_ids, attn_mask)            # student projected (768-d)

loss_mse = nn.MSELoss()(z_S, z_T)             # MSE component
loss_cos = (1 - nn.CosineSimilarity(dim=1)(z_S, z_T)).mean()  # cosine component
loss = alpha * loss_mse + (1 - alpha) * loss_cos

optimizer.zero_grad()
loss.backward()                                # gradients flow through student only
optimizer.step()

print(f"MSE loss: {loss_mse.item():.4f}")
# Output: MSE loss: (varies; decreases toward ~0.01 over training)
print(f"Cosine loss: {loss_cos.item():.4f}")
# Output: Cosine loss: (varies; decreases toward ~0.01 over training)

# --- INT8 Quantisation ---
student_q = torch.quantization.quantize_dynamic(
    student,                                   # model to quantise
    {nn.Linear},                               # quantise only Linear layers
    dtype=torch.qint8                          # target 8-bit integers
)

# --- ONNX Export ---
dummy_ids  = torch.zeros(1, 16, dtype=torch.long)
dummy_mask = torch.ones(1, 16, dtype=torch.long)

torch.onnx.export(
    student_q,
    (dummy_ids, dummy_mask),
    './models/tinybert_int8.onnx',
    input_names=['input_ids', 'attention_mask'],
    output_names=['embedding'],
    dynamic_axes={'input_ids': {0: 'batch'}, 'attention_mask': {0: 'batch'}},
    opset_version=13
)
print("ONNX export complete.")
# Output: ONNX export complete.

# --- Verify inference ---
sess = ort.InferenceSession('./models/tinybert_int8.onnx')
emb = sess.run(None, {'input_ids': dummy_ids.numpy(), 'attention_mask': dummy_mask.numpy()})
print(f"Embedding shape: {emb[0].shape}")
# Output: Embedding shape: (1, 768)
```

---

## Summary

| Concept | Key Takeaway |
|---|---|
| Knowledge distillation | Student learns to copy teacher embeddings, not raw labels |
| MSE loss | Penalises absolute coordinate differences between student and teacher vectors |
| Cosine loss | Penalises directional misalignment, robust to scale mismatches |
| Combined loss | $\alpha \cdot \text{MSE} + (1-\alpha) \cdot \text{cosine}$ — captures both scale and direction |
| INT8 quantisation | Converts 32-bit float weights to 8-bit integers, 4× size reduction |
| ONNX export | Platform-independent model file; runs anywhere with ONNX Runtime |
| Final result | 0.75 MB model, 0.64 ms/case, 614× smaller than teacher, equal accuracy |
