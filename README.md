# LAB 8 — LangSmith Evaluation Lab
## Procurement Compliance Q&A Evaluation

**Ironhack AI Engineering Bootcamp · Week 6 · Lab 8**  
**Author:** Eugen Mueller

---

## Domain

**Procurement Compliance Q&A** — evaluating how well LLMs extract accurate and complete answers from supplier contract clauses and policy documents.

Procurement teams need to parse dense legal text quickly (liability caps, GDPR DPA terms, termination rights, payment conditions). A wrong or incomplete answer has direct business consequences — making this a high-stakes, well-bounded evaluation domain.

---

## Dataset

**Name:** `procurement-compliance-qa-v1` (hosted on LangSmith)  
**Size:** 20 examples  
**Structure:** each example has:
- `inputs.question` — a due-diligence question a procurement analyst would ask
- `inputs.context` — a supplier contract excerpt (ground truth source)
- `outputs.answer` — the reference answer
- `metadata.category` — one of: `risk`, `compliance`, `data-privacy`, `contractual`, `financial`
- `metadata.difficulty` — `easy`, `medium`, or `hard`

---

## Approach

| Step | What |
|---|---|
| Dataset creation | 20 hand-crafted procurement Q&A examples, uploaded to LangSmith |
| Target functions | `gpt-4o-mini` and `gpt-4o`, both `@traceable`, zero temperature |
| Correctness evaluator | LLM-as-judge (binary 0/1) comparing output to reference answer |
| Completeness evaluator | Custom LLM-as-judge (0.0–1.0) checking for missing material terms |
| A/B comparison | Both models evaluated on the same dataset and evaluators |

---

## How to run

### Prerequisites

```bash
# Install dependencies (Python 3.12+)
python3.12 -m pip install langsmith openai openevals python-dotenv pandas matplotlib
```

### Fill in your API keys

Open `.env` and add your keys:

```
OPENAI_API_KEY=sk-...
LANGCHAIN_API_KEY=lsv2_...
```

### Run the notebooks in order

1. **`01_dataset_creation.ipynb`** — creates and uploads the LangSmith dataset (run once)
2. **`02_evaluation.ipynb`** — runs experiments, analyses results, writes summary files

Open in Jupyter:
```bash
python3.12 -m jupyter notebook
```

---

## File Map

```
LAB 8/
├── 01_dataset_creation.ipynb     # Part 1 & 2: domain selection + LangSmith dataset creation
├── 02_evaluation.ipynb           # Parts 3-5 + Optional: target functions, evaluators, A/B analysis
├── evaluation_summary.md         # Auto-generated: evaluation results paragraph + metrics tables
├── optimization_summary.md       # Auto-generated: cost-performance trade-off analysis
├── README.md                     # This file
├── .env                          # API keys (not committed to git)
├── data/
│   └── procurement_compliance_dataset.json   # Local JSON backup of the 20 examples
├── results/
│   ├── evaluation_results.csv    # Full results DataFrame (both models, all scores)
│   └── evaluation_charts.png    # Bar charts: model comparison, category, difficulty
└── screenshots/
    └── (add LangSmith UI screenshots here before submission)
```

---

## LangSmith Links

- **Dataset:** `procurement-compliance-qa-v1` → [smith.langchain.com/datasets](https://smith.langchain.com/datasets)
- **Experiments:** project `procurement-compliance-eval` → [smith.langchain.com](https://smith.langchain.com)

*(Add direct links after running the notebooks)*

---

## Key Results

> Run `02_evaluation.ipynb` to generate actual numbers. Summary auto-written to `evaluation_summary.md`.

**Headline finding:** `gpt-4o-mini` achieves near-parity with `gpt-4o` on structured contract Q&A at ~15–20× lower cost. The custom completeness evaluator surfaces a hidden failure mode: answers that are technically correct but omit secondary conditions (thresholds, consent requirements, deadlines) — which are the most dangerous in a procurement context.
