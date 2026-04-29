---
title: Clinical-Lightweight-Language-Model
colorFrom: blue
colorTo: indigo
sdk: docker
---

<div align="center">

<h1>🏥 Clinical Lightweight Language Model</h1>
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=3b82f6&center=true&vCenter=true&width=700&lines=Surgical+Procedure+Duration+Prediction;TinySurgicalBERT+%C2%B7+0.7+MB+%C2%B7+Mobile-Ready;178%2C512+Cases+%C2%B7+8+Regression+Models" alt="Typing SVG"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3b82f6?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-4f46e5?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar/spaces)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-3b82f6?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)](#)

<br/>

**🏥 Clinical Lightweight Language Model** — Surgical procedure duration prediction using a domain-specific distilled BERT model (TinySurgicalBERT), evaluated against Bio-ClinicalBERT and SentenceBERT baselines across eight regression models on 178,512 surgical cases.

<br/>

---

</div>

## Table of Contents

- [Features](#-features)
- [Architecture](#️-architecture)
- [Getting Started](#-getting-started)
- [Pipeline Stages](#-pipeline-stages)
- [ML Models](#-ml-models)
- [Project Structure](#-project-structure)
- [Author](#-author)
- [Contributing](#-contributing)
- [Disclaimer](#disclaimer)
- [License](#-license)

---

## ✨ Features

<table>
  <tr>
    <td>🧠 <b>TinySurgicalBERT</b></td>
    <td>Domain-distilled BERT (0.7 MB ONNX) — mobile-ready surgical text encoder trained from BioBERT</td>
  </tr>
  <tr>
    <td>📊 <b>Multi-Baseline Evaluation</b></td>
    <td>Benchmarked against Bio-ClinicalBERT and SentenceBERT across eight regression algorithms</td>
  </tr>
  <tr>
    <td>🔁 <b>Five-Stage Distillation</b></td>
    <td>Progressive knowledge transfer (D1–D5) from BioBERT teacher to lightweight student model</td>
  </tr>
  <tr>
    <td>📈 <b>Large-Scale Validation</b></td>
    <td>178,512 real surgical cases with stratified 5-fold cross-validation</td>
  </tr>
  <tr>
    <td>🔬 <b>Reproducible Pipeline</b></td>
    <td>End-to-end scripts for distillation, embedding, regression, and figure generation</td>
  </tr>
  <tr>
    <td>📄 <b>Manuscript-Ready Outputs</b></td>
    <td>LaTeX tables and PDF figures written directly to <code>results/</code> and <code>overleaf/</code></td>
  </tr>
</table>

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│              Clinical Lightweight Language Model                  │
│                                                                    │
│  ┌────────────┐    ┌─────────────────┐    ┌──────────────────┐  │
│  │  Surgical  │───▶│ TinySurgical    │───▶│ Regression Models│  │
│  │   Data     │    │    BERT         │    │  (8 algorithms)  │  │
│  └────────────┘    └───────┬─────────┘    └────────┬─────────┘  │
│                            │                        │              │
│                   ┌────────▼────────┐      ┌────────▼────────┐   │
│                   │  Distillation   │      │   Predictions   │   │
│                   │   (D1 – D5)     │      │  (Duration, min)│   │
│                   └─────────────────┘      └─────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Git
- `data/raw/surgical_data.db` (source database — not included in repo)

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/mnoorchenar/Clinical-Lightweight-Language-Model.git
cd Clinical-Lightweight-Language-Model

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Reproduce Results

```bash
# Step 1 — Distil TinySurgicalBERT (requires data/raw/surgical_data.db)
python code/distill_bert.py

# Step 2 — Run the full experiment pipeline
python code/pipeline.py

# Step 3 — Generate figures and LaTeX tables
python code/figures.py
```

All outputs are written to `data/outputs/` (metrics, models) and `results/` (figures, tables).

### Compile Manuscript

```bash
pdflatex -interaction=nonstopmode -jobname=overleaf/main overleaf/00_main.tex
```

Output: `overleaf/main.pdf`

---

## 📊 Pipeline Stages

| Stage | Description | Status |
|-------|-------------|--------|
| 🔤 D1 Tokenizer Training | Domain-adaptive tokenizer fine-tuning on surgical corpus | ✅ Complete |
| 🧩 D2 MLM Pre-training | Masked language modelling to acquire surgical semantics | ✅ Complete |
| 📐 D3 Embedding Distillation | Layer-wise teacher-to-student knowledge transfer | ✅ Complete |
| 📦 D4 ONNX Export | Quantised, mobile-ready export (0.7 MB) | ✅ Complete |
| ✔️ D5 Validation | Embedding quality and downstream task evaluation | ✅ Complete |
| 🔬 01–04 Experiment | Full regression pipeline, cross-validation, and benchmarking | ✅ Complete |

---

## 🧠 ML Models

```python
# Models evaluated in Clinical Lightweight Language Model
encoders = {
    "TinySurgicalBERT": "Distilled BioBERT (0.7 MB ONNX) — proposed model",
    "Bio-ClinicalBERT":  "Clinical domain BERT baseline",
    "SentenceBERT":      "Sentence-level embedding baseline",
}

regressors = {
    "Ridge":       "Linear regression with L2 regularisation",
    "Lasso":       "Linear regression with L1 regularisation",
    "SVR":         "Support Vector Regression",
    "RandomForest":"Ensemble of decision trees",
    "XGBoost":     "Gradient-boosted trees (XGBoost)",
    "LightGBM":    "Gradient-boosted trees (LightGBM)",
    "GBM":         "Gradient Boosting Machine",
    "MLP":         "Multi-layer perceptron",
}
```

---

## 📁 Project Structure

```
Clinical-Lightweight-Language-Model/
│
├── 📂 code/
│   ├── 📄 distill_bert.py     # Knowledge distillation pipeline (Stages D1–D5)
│   ├── 📄 pipeline.py         # Main experiment pipeline (Stages 01–04)
│   └── 📄 figures.py          # Figure and table generation → results/
│
├── 📂 data/
│   ├── 📂 raw/                # Source data — read only, never modify
│   ├── 📂 processed/          # Preprocessed features, fold assignments, embedding cache
│   └── 📂 outputs/            # Result database, benchmark arrays, trained model artefacts
│
├── 📂 results/                # Generated figures (.pdf) and LaTeX tables (.tex)
│
├── 📂 overleaf/               # LaTeX manuscript (Elsevier elsarticle)
│   ├── 📄 00_main.tex         # Master file
│   ├── 📄 01_abstract.tex
│   ├── 📄 02_abbreviations.tex
│   ├── 📄 03_introduction.tex
│   ├── 📄 04_related_works.tex
│   ├── 📄 05_methodology.tex
│   ├── 📄 06_flowchart.tex
│   ├── 📄 07_results.tex
│   ├── 📄 08_conclusion.tex
│   ├── 📄 bibliography.tex    # \bibliographystyle + \bibliography
│   ├── 📄 references.bib      # Consolidated bibliography
│   ├── 📂 introduction/       # Reference PDFs + introduction.bib
│   ├── 📂 related_works/      # related_works.bib (staging area)
│   └── 📄 000_search_requests.md
│
├── 📂 docs/                   # Internal notes and methodology documentation
├── 📄 prompt.md               # Claude Code session prompt
└── 📄 sync.ps1                # Git sync utility
```

---

## 👨‍💻 Author

<div align="center">

<table>
<tr>
<td align="center" width="100%">

<img src="https://avatars.githubusercontent.com/mnoorchenar" width="120" style="border-radius:50%; border: 3px solid #4f46e5;" alt="Mohammad Noorchenarboo"/>

<h3>Mohammad Noorchenarboo</h3>

<code>Data Scientist</code> &nbsp;|&nbsp; <code>AI Researcher</code> &nbsp;|&nbsp; <code>Biostatistician</code>

📍 &nbsp;Ontario, Canada &nbsp;&nbsp; 📧 &nbsp;[mohammadnoorchenarboo@gmail.com](mailto:mohammadnoorchenarboo@gmail.com)

──────────────────────────────────────

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mnoorchenar)&nbsp;
[![Personal Site](https://img.shields.io/badge/Website-mnoorchenar.github.io-4f46e5?style=for-the-badge&logo=githubpages&logoColor=white)](https://mnoorchenar.github.io/)&nbsp;
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar/spaces)&nbsp;
[![Google Scholar](https://img.shields.io/badge/Scholar-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white)](https://scholar.google.ca/citations?user=nn_Toq0AAAAJ&hl=en)&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar)

</td>
</tr>
</table>

</div>

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

---

## Disclaimer

<span style="color:red">This project is developed strictly for educational and research purposes and does not constitute professional medical advice of any kind. All datasets used are subject to institutional data-use agreements — no patient-identifiable information is included in this repository. This software is provided "as is" without warranty of any kind; use at your own risk.</span>

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:3b82f6,100:4f46e5&height=120&section=footer&text=Made%20with%20%E2%9D%A4%EF%B8%8F%20by%20Mohammad%20Noorchenarboo&fontColor=ffffff&fontSize=18&fontAlignY=80" width="100%"/>

[![GitHub Stars](https://img.shields.io/github/stars/mnoorchenar/Clinical-Lightweight-Language-Model?style=social)](https://github.com/mnoorchenar/Clinical-Lightweight-Language-Model)
[![GitHub Forks](https://img.shields.io/github/forks/mnoorchenar/Clinical-Lightweight-Language-Model?style=social)](https://github.com/mnoorchenar/Clinical-Lightweight-Language-Model/fork)

<sub>This project is developed purely for academic and research purposes. Any similarity to existing company names, products, or trademarks is entirely coincidental and unintentional. This project has no affiliation with any commercial entity.</sub>

</div>
