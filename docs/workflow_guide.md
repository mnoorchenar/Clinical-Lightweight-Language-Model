# Paper Writing Workflow — Complete Guide

This document explains exactly what to do at every step of the paper writing pipeline,
with special focus on the two steps that require you to find and provide papers:
Step 5 (Related Works) and Step 6 (Introduction).

---

## Part 1 — The Big Picture

### What the 9 Steps Are

The pipeline has 9 steps. They run in order. You never skip forward.

| Step | What Gets Written | Do You Need Papers? |
|------|-------------------|---------------------|
| 1 — Methodology | 05_methodology.tex + 06_flowchart.tex | No |
| 2 — Preprocessing | Python code in code/ | No |
| 3 — Experiments | Python code + figures in results/ | No |
| 4 — Results | 07_results.tex | No |
| 5 — Related Works | 04_related_works.tex | **Yes — cluster by cluster** |
| 6 — Introduction | 03_introduction.tex | **Yes — all at once** |
| 7 — Conclusion | 08_conclusion.tex | No |
| 8 — Abstract + Cleanup | 01_abstract.tex + checklist | No |
| 9 — Documentation | docs/ files | No (only if requested) |

Steps 1–4 are purely technical. Claude writes code and LaTeX based on your data and
results — you do not need to find any papers for these steps.

Steps 5 and 6 are where papers matter. The rule is: **Claude never writes a sentence
that needs a citation until you have provided the paper that supports it.**

This is called Mode A (papers-first). It exists because if Claude writes first and
you search for papers later, you will find that many sentences in the draft cannot
be matched to a real paper — or the paper says something slightly different. Writing
from real papers eliminates this problem entirely.

---

## Part 2 — The Two Files You Need to Know

Before explaining Step 5 and Step 6, you need to understand two files that manage
all paper searching.

### File 1: overleaf/000_search_requests.md

**What it is:** A tracking file that Claude maintains. It tells you exactly what to
search for and whether each cluster is done or still needs papers.

**When to read it:** At the start of Step 5 and Step 6. It will show you which clusters
still have "NEED PAPERS" status.

**You never edit this file.** Claude updates it when a cluster is completed.

**Example of what it looks like:**

```
### Cluster 3 — Clinical NLP and BERT for medical text
Status: NEED PAPERS
Search:  clinical BERT medical text NLP
Add results to: overleaf/paper_selected.md
Add BibTeX to:  overleaf/related_works/related_works.bib
```

This tells you: go to Google Scholar, search `clinical BERT medical text NLP`,
pick 3–5 papers, paste their details into paper_selected.md, and add their
BibTeX entries to related_works.bib.

---

### File 2: overleaf/paper_selected.md (or related_works/paper_selected.md)

**What it is:** A file where you paste the papers you find. Claude reads this file
before writing each section.

**When to edit it:** After each Google Scholar search, before asking Claude to write
the corresponding cluster or paragraph.

**Format to use** (copy exactly):

```
### Cluster 3 — clinical BERT medical text NLP

3. Fine-tuning BERT on electronic health record notes
   Authors: F Li, Y Jin, W Liu - JMIR Medical Informatics, 2019
   Summary: Fine-tunes BioBERT on 1.5 million EHR notes; shows domain-adapted
            BERT substantially improves clinical entity normalisation vs general BERT.
   Citations: 219
```

The Summary is the most important field. Write what the paper actually says that
is relevant to your argument — the specific claim or statistic you will cite.

---

## Part 3 — Two Types of Searches

When Claude generates a search request, it will always be one of two types.

### Type 1 — Full Paper Title (for known papers)

Used when the paper is a well-known landmark that Claude already knows about —
for example, the original BERT paper, DistilBERT, or XGBoost.

**What it looks like in 000_search_requests.md:**
```
Search:  Attention Is All You Need
```

**What you do:** Paste the exact title into Google Scholar. The paper will appear
immediately. Download the PDF and copy the BibTeX entry.

**Why this works:** These papers are so well-known that searching the full title
gives you the exact paper on the first result.

---

### Type 2 — Short Keyword (for papers Claude does not know)

Used when the topic is known but the specific papers are not — for example,
"papers about AI deployment in operating rooms."

**What it looks like in 000_search_requests.md:**
```
Search:  clinical BERT medical text NLP
```

**What you do:** Paste the keyword into Google Scholar. Browse the first 10–15
results. Pick 3–5 papers that are most relevant to your paper's argument.
You choose — Claude does not decide which papers to select.

**How to pick good papers:**
- Prefer papers with more citations (more established)
- Prefer papers that directly address the gap your paper fills
- Prefer papers from journals (not just conference proceedings) when possible
- If two papers say the same thing, keep only the more cited one

**What you do NOT need to do:** Read the full papers. The abstract and the summary
you write in paper_selected.md is enough for Claude to write from.

---

## Part 4 — Step 5 (Related Works) in Detail

### What the Related Works section does

The Related Works section reviews all prior work that is relevant to your paper,
organises it into thematic clusters (groups by topic), and ends with a paragraph
explaining exactly how your paper is different from and better than all reviewed work.

### The cluster structure

The Related Works section is divided into 3–4 subsections called clusters. Each
cluster covers one topic area. For a paper about TinySurgicalBERT, the clusters are:

| Cluster | Topic | What it argues |
|---------|-------|----------------|
| 1 | ML for surgical duration prediction | Prior ML uses structured data only — ignores procedure text |
| 2 | AI for OR management | Focuses on accuracy/interpretability — ignores deployment constraints |
| 3 | Clinical language models | Good at text but too large for edge hardware |
| 4 | Knowledge distillation | Compression exists but never applied to surgical text |

Each cluster ends by identifying its gap. The final paragraph synthesises all four
gaps into one argument: "This paper fills the intersection that none of the above reaches."

### Your role in Step 5 — exact sequence

**Step 5a — Claude gives you the cluster plan and search keywords.**
Claude opens 000_search_requests.md and fills in one keyword per cluster. At this point
the section is not written yet. Example output from Claude:

```
Cluster 1 — ML for surgical duration: bartek2019, zhao2019, martinez2021 (DONE)
Cluster 2 — OR management: bellini2024, bellini2020, dios2015 (DONE)
Cluster 3 — Clinical NLP: NEED PAPERS → Search: clinical BERT medical text NLP
Cluster 4 — Knowledge distillation: NEED PAPERS → Search: knowledge distillation BERT compression
```

**Step 5b — You search and provide papers for the clusters that need them.**
For each "NEED PAPERS" cluster:
1. Open Google Scholar
2. Search the keyword exactly as written
3. Browse results, pick 3–5 papers
4. Paste into paper_selected.md (using the format shown in Part 2)
5. Add BibTeX entries to overleaf/related_works/related_works.bib

**Step 5c — Tell Claude you have added the papers.**
Say something like: "I added papers for clusters 3 and 4 — please check and write."

**Step 5d — Claude reviews what you provided, removes irrelevant papers, and writes.**
Claude reads your paper_selected.md, decides which papers to keep (and tells you
which it removed and why), then writes the full Related Works section using only
real citations — no [CITE] placeholders.

**Step 5e — Claude compiles the PDF.**
After writing, Claude always runs the full compile. You review the PDF.

---

### What makes a paper "not relevant" to Related Works

Claude removes a paper if:
- It covers a completely different task (e.g., a paper about radiology image analysis
  when the cluster is about surgical text)
- It duplicates another paper already in the cluster (same authors, same year,
  same point — keep only the stronger one)
- It is from a paradigm too different to be contrasted usefully (e.g., symbolic AI
  when the cluster is about neural compression)

When Claude removes a paper, it always explains why. You can disagree and ask
Claude to keep it.

---

## Part 5 — Step 6 (Introduction) in Detail

### What the Introduction does

The Introduction builds the argument that leads the reader from "this is an important
problem" to "this paper is the right solution." It does this across exactly 7 paragraphs:

| Paragraph | Topic | Key argument |
|-----------|-------|--------------|
| P1 | OR scheduling consequences | Errors cause overtime, cancellations, patient harm |
| P2 | Economic cost of OR | OR time costs $46/min; errors cost billions |
| P3 | ML with structured EHR | ML helps but ignores free-text procedure descriptions |
| P4 | Free-text adds value | Adding procedure text reduces MAE by 16% |
| P5 | Clinical LMs and their cost | BERT variants work but are 100–440 MB — not deployable |
| P6 | Knowledge distillation | KD can compress BERT but not yet for surgical text |
| P7 | Contribution | TinySurgicalBERT fills all three gaps; paper structure |

### Why Introduction is different from Related Works

In Related Works, papers are discovered through searching — you do not know in advance
which papers will be relevant. In the Introduction, most papers are already known.
BERT, DistilBERT, Bio-ClinicalBERT, and SentenceBERT are landmark papers with fixed
arXiv IDs. For these, you search by full title, not by keyword.

The Introduction therefore works slightly differently:

- **Known papers** (P5, P6): Claude tells you the exact title. You search it directly.
  These are typically already in introduction.bib from a previous session.
- **Domain papers** (P1–P4): You may need to search by keyword if the specific papers
  are not yet in introduction.bib.

### Your role in Step 6 — exact sequence

**Step 6a — Check 000_search_requests.md.**
The Introduction section will list which papers are already available (from
introduction.bib) and which still need to be found. If the introduction was already
written (as in the current TinySurgicalBERT project), all papers are listed as DONE.

**Step 6b — For any paragraph that needs papers, search and provide.**
If a paragraph shows "NEED PAPERS", do the same as Step 5b: search, pick, paste
into paper_selected.md, add BibTeX to introduction.bib.

**Step 6c — Tell Claude all papers are ready.**
Claude then writes the full 7-paragraph Introduction with real citations only.

**Step 6d — Review the contribution paragraph (P7) carefully.**
P7 is the most important paragraph. It should:
- State what was built and why it matters (plain language — no math, no stage names)
- State how it was evaluated and the headline result
- End with the paper structure sentence

It should NOT contain methodology detail. If you see loss function notation,
stage names, or sentences that describe how something works — those belong in
the Methodology section, not here.

---

## Part 6 — How Introduction and Related Works Connect

The Introduction and Related Works are written to complement each other, not repeat
each other. Their roles are strictly separated:

| Location | Role | Example |
|----------|------|---------|
| Introduction P1–P6 | Build the problem → solution argument | "Bio-ClinicalBERT is 436 MB, making it impractical for edge deployment" |
| Introduction P7 | State what this paper proposes (high level) | "TinySurgicalBERT achieves equivalent accuracy at 580× less storage" |
| Related Works subsections | Review each prior cluster in depth | Compare 5 papers on knowledge distillation, note what each lacks |
| Related Works final paragraph | Position this paper against all reviewed work | "No prior work applies KD to the surgical domain with this paper filling that gap" |
| Methodology opening | Explain how the solution was built (technical) | "A 2-layer student with 128-d hidden size was trained using MSE + cosine loss" |

**The key rule:** If you read Introduction P7 and Related Works final paragraph back
to back, they should not feel repetitive. P7 says what the paper does; the Related
Works final paragraph says how it differs from specific prior work reviewed in that section.

---

## Part 7 — Quick Reference Card

### At the start of each session

Tell Claude: "Read prompt.md. Today we are working on Step X."

### When starting Step 5 (Related Works)

1. Ask Claude to show the cluster plan and search keywords
2. Open 000_search_requests.md — read each cluster's status
3. For each "NEED PAPERS" cluster: search on Google Scholar, paste 3–5 papers into paper_selected.md
4. Add BibTeX entries to overleaf/related_works/related_works.bib
5. Tell Claude: "I added papers for clusters X and Y"
6. Claude reviews, removes irrelevant ones, writes the section, compiles

### When starting Step 6 (Introduction)

1. Open 000_search_requests.md — check Introduction section status
2. If all papers are DONE: tell Claude "All papers are ready, write the introduction"
3. If any paragraph shows NEED PAPERS: search using the keyword or full title provided,
   paste into paper_selected.md, add BibTeX to introduction.bib
4. Claude writes 7 paragraphs, all real citations, no placeholders, compiles

### Search keyword format

| Situation | What to search | Example |
|-----------|----------------|---------|
| Claude gives a full title | Paste exact title into Google Scholar | `Attention Is All You Need` |
| Claude gives a short keyword | Paste keyword into Google Scholar, browse first 15 results | `knowledge distillation BERT compression` |

### BibTeX goes where

| Section | BibTeX file |
|---------|-------------|
| Introduction papers | overleaf/introduction/introduction.bib |
| Related Works papers | overleaf/related_works/related_works.bib |
| Papers cited in both | Either file — BibTeX only needs to be in one place |

---

## Part 8 — Common Mistakes to Avoid

**Mistake 1: Starting to write before providing papers.**
Always provide papers first. If you ask Claude to write and some papers are missing,
Claude will use [CITE: ...] placeholders — which you will then need to fill in later,
defeating the purpose of Mode A.

**Mistake 2: Providing too many papers.**
3–5 papers per cluster is ideal. More than 6 creates a summary paragraph that is too
broad and loses argumentative clarity. Claude will remove extras anyway — save yourself
the work and be selective when searching.

**Mistake 3: Providing papers that are off-topic.**
Read the cluster name before searching. Cluster 4 is "Knowledge distillation and compact
transformers." A paper about neural network pruning for image classification is not
relevant to this cluster — even though it mentions compression. The paper must be
about text model compression or it will be removed.

**Mistake 4: Forgetting to add BibTeX.**
paper_selected.md without a BibTeX entry in the .bib file means Claude cannot write
`\cite{key}` — it has no key to reference. Always add both at the same time.

**Mistake 5: Confusing the Introduction and Related Works paper workflows.**
- Introduction: most papers are known in advance (landmarks like BERT, DistilBERT)
  → search by full title → add to introduction.bib
- Related Works: papers are discovered through search → search by keyword → add to related_works.bib

---

## Part 9 — Status of the Current TinySurgicalBERT Project

For reference, here is the current state of both sections in this project:

### Introduction — COMPLETE
All 11 papers are in introduction.bib. The section is written and compiled.
No further paper searching is needed for the introduction.

Papers used: oliveira2023systematic, park2025machine, al2025comprehensive,
wang2022more, kwong2025optimizing, noorchenarboo2026benchmarking,
devlin2019bert, reimers2019sentence, alsentzer2019publicly,
sanh2019distilbert, jiao2020tinybert

### Related Works — COMPLETE
All 4 clusters are written and compiled. 15 papers are in related_works.bib.
No further paper searching is needed for related works.

Cluster 1 (ML prediction): martinez2021machine, bartek2019improving, zhao2019machine,
stepaniak2009modeling, stepaniak2010modeling

Cluster 2 (OR management): bellini2024artificial, bellini2020artificial, dios2015decision

Cluster 3 (Clinical NLP): li2019fine, nunes2024health, turchin2023comparison

Cluster 4 (KD compression): sun2019patient, gupta2022compression, tang2024survey, bibi2024advances

### Next step for this project: Step 8 — Abstract and Title (already drafted), then final cleanup checklist.
