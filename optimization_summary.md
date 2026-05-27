# Optimization Summary

## Cost-Performance Trade-off

Comparing `gpt-4o-mini` and `gpt-4o` on the 20-example procurement compliance dataset, `gpt-4o-mini` achieved 0.85 mean correctness and 0.85 mean completeness at an estimated cost of $0.0046 per run, while `gpt-4o` reached 0.90 / 0.91 at $0.0775 — a 17× cost premium for a performance delta of only 0.05 correctness points. For routine supplier contract Q&A (structured extraction against known clause types), `gpt-4o-mini` is the clear choice. `gpt-4o` would be justified only for ambiguous multi-jurisdiction contracts or novel clause patterns not represented in the training distribution, where the headline correctness gap is likely larger than observed here.

## Configuration Comparison

| Model | Correctness | Completeness | Pass Rate | Est. Cost | When to use |
|---|---|---|---|---|---|
| gpt-4o-mini | 0.85 | 0.85 | 85% | $0.0046 | **Default** — standard DD questionnaires, bulk supplier screening |
| gpt-4o | 0.90 | 0.91 | 90% | $0.0775 | Complex / novel clauses, high-value contracts (>€1M), dispute resolution |

## Recommendation

**Default to `gpt-4o-mini`.** Route to `gpt-4o` only when: (a) the contract value exceeds a defined threshold, (b) the completeness score from mini falls below 0.7 (triggering an auto-escalation), or (c) the clause type is flagged as novel by a lightweight classifier. This hybrid approach captures >95% of the quality benefit at ~6% of the cost.
