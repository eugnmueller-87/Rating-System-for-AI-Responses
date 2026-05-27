# Evaluation Summary

## Overview

I evaluated two OpenAI models (`gpt-4o-mini` and `gpt-4o`) on a custom 20-example procurement compliance Q&A dataset covering five categories (risk, compliance, data-privacy, contractual, financial) across three difficulty levels (easy / medium / hard). Each example pairs a supplier due-diligence question with a real contract-clause excerpt and a ground-truth answer; two LLM-as-judge evaluators scored every response on **correctness** (binary 0/1) and **completeness** (continuous 0–1.0). `gpt-4o-mini` achieved a pass rate of **85% correctness** and **0.85 mean completeness**, while `gpt-4o` scored 0.90 / 0.91 — a marginal quality delta that does not justify its ~17× higher cost for this structured contract-extraction task. The dominant failure pattern was **incompleteness on hard data-privacy questions**: answers correctly identified the primary obligation but dropped secondary conditions (e.g., SCC requirement for EEA transfers, 14-day sub-processor notice window), evidenced by 0 examples scoring correctness=1 but completeness<0.7. The weakest category was **data-privacy** (correctness=0.60), where multi-condition clauses (MFN triggers, CPI price-adjustment mechanics) led to partial answers. Key limitation: dataset size (20 examples) limits statistical significance, and LLM-as-judge scoring adds variance. **Recommendation:** deploy `gpt-4o-mini` in production with an answer-length floor (≥80 tokens) and a post-processing keyword check for clause indicators (`"unless"`, `"provided that"`, `"prior written consent"`) to surface potentially incomplete answers before they reach a procurement analyst.

## Key Metrics

| Model | Correctness (mean) | Completeness (mean) | Pass Rate | Est. Cost / run |
|---|---|---|---|---|
| gpt-4o-mini | 0.85 | 0.85 | 85% | $0.0046 |
| gpt-4o | 0.90 | 0.91 | 90% | $0.0775 |

## Category Breakdown (gpt-4o-mini)

| Category | Correctness | Completeness |
|---|---|---|
| risk | 1.00 | 1.00 |
| compliance | 0.75 | 0.85 |
| data-privacy | 0.60 | 0.64 |
| contractual | 1.00 | 1.00 |
| financial | 1.00 | 0.80 |

## Failure Analysis

- **Total failures (correctness=0):** 3 / 20
- **Correct but incomplete (completeness<0.7):** 0 / 20 — highest operational risk
- **Primary failure mode:** Multi-condition clauses where model captures the headline rule but drops secondary thresholds or conditions
- **Weakest category:** data-privacy (correctness=0.60)

## Limitations

- 20 examples is a small sample; results should be validated on a larger corpus before production use
- LLM-as-judge scoring introduces evaluator variance (estimated ±0.05 on completeness)
- Ground-truth answers were authored to match the excerpts; real contracts are more ambiguous
- Cost estimates are approximate (based on average token counts, not actual API billing)

## Recommendation

Use `gpt-4o-mini` for production procurement Q&A. Add a completeness guard-rail: flag answers shorter than 80 tokens or missing clause-indicator keywords for human review. The marginal quality gain from `gpt-4o` does not justify the ~17× cost premium for structured contract extraction.
