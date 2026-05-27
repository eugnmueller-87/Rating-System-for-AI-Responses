# Evaluation Summary

## Overview

I evaluated two OpenAI models (`gpt-4o-mini` and `gpt-4o`) on a custom 20-example procurement compliance Q&A dataset covering five categories (risk, compliance, data-privacy, contractual, financial) across three difficulty levels. Each example pairs a supplier due-diligence question with a real contract-clause excerpt and a ground-truth answer; two LLM-as-judge evaluators scored every response on **correctness** (binary 0/1) and **completeness** (continuous 0-1.0). `gpt-4o-mini` achieved **80% correctness** and **0.86 mean completeness**, while `gpt-4o` scored 0.90 / 0.89 — a marginal delta that does not justify its ~17x higher cost. The weakest category was **data-privacy** (correctness=0.60), where multi-condition clauses led to partial answers. **Recommendation:** use `gpt-4o-mini` with an answer-length floor and keyword completeness check before answers reach a procurement analyst.

## Key Metrics

| Model | Correctness | Completeness | Pass Rate | Est. Cost |
|---|---|---|---|---|
| gpt-4o-mini | 0.80 | 0.86 | 80% | $0.0046 |
| gpt-4o | 0.90 | 0.89 | 90% | $0.0775 |

## Category Breakdown (gpt-4o-mini)

| Category | Correctness | Completeness |
|---|---|---|
| risk | 1.00 | 1.00 |
| compliance | 0.75 | 0.85 |
| data-privacy | 0.60 | 0.82 |
| contractual | 0.75 | 0.77 |
| financial | 1.00 | 0.90 |

## Failure Analysis

- **Failures (correctness=0):** 4 / 20
- **Correct but incomplete (<0.7):** 0 / 20
- **Weakest category:** data-privacy (0.60)

## Limitations

- 20 examples limits statistical significance
- LLM-as-judge adds scoring variance (~±0.05)
- Cost estimates are approximate

## Recommendation

Use `gpt-4o-mini`. Route to `gpt-4o` only for high-value or ambiguous contracts. The ~17x cost premium is not justified for structured contract Q&A.
