# Natural Isoform Failure Audit

Source artifacts:

- `tmp/hoptf_guardrails_may18/extracted/tmp/hopfield_fitting_leave_one_panel_3seed_calibrated/otcfm_leave_one_panel_aggregate.csv`
- `tmp/hoptf_guardrails_may18/extracted/tmp/hopfield_fitting_leave_one_panel_3seed_calibrated/otcfm_leave_one_panel_metrics.csv`
- Compact derived table: `tmp/hoptf_guardrails_may18/natural_isoform_failure_audit_table.csv`

## Summary

The calibrated mixed-response same-gene isoform panel is useful but not clean enough to treat as full endpoint validation. It produced `8 / 14` stable label-aware passes across three seeds. The remaining six non-stable isoforms split into one fair responder failure, one unsupported nonresponder case, and four seed-sensitive or threshold-sensitive cases.

This supports using mutant transport as a preliminary sequence-sensitivity result, not as a standalone validation of the endpoint model. The report should explicitly state that the endpoint model is still imperfect on natural same-gene isoform holdouts.

## Failure Audit

| Isoform | Label | Key evidence | Audit classification | Report interpretation |
| --- | --- | --- | --- | --- |
| `IKZF3-5` | responder | `0 / 3` label-aware passes; mean control-standardized endpoint MSE fraction `1.175`; predicted response is only `0.20x` observed response on average; same-gene sibling support includes `3` responder and `2` nonresponder siblings. | Fair model failure, with low-cell caveat. | This is the clearest natural-isoform limitation. The model under-transports a real responder despite sibling support. |
| `SOX5-1` | nonresponder | `0 / 3` label-aware passes; no same-gene nonresponder sibling support; response ratio is very high (`7.93x` mean), indicating over-transport. | Unsupported nonresponder holdout. | Do not treat as a clean biological failure. It shows the panel can generate unfair nonresponder tests when same-label sibling support is absent. |
| `ZNF195-3` | responder | `1 / 3` label-aware passes; mean endpoint MSE fraction `1.066`; predicted response is very low (`0.073x` observed response); `7` recovered cells. | Seed-sensitive under-transport. | Useful caution that strong low-cell responders remain unstable. Not a hard failure because one seed passes and endpoint error is near the threshold. |
| `ZNF534-4` | responder | `1 / 3` label-aware passes; mean endpoint MSE fraction `0.960`, but two seeds fail; only `1` responder sibling. | Seed-sensitive with weak responder support. | Treat as instability rather than a decisive biological failure. Mean endpoint performance is near acceptable, but seed behavior is not stable. |
| `TP73-1` | responder | `2 / 3` label-aware passes; mean endpoint MSE fraction `0.941`; failed stable threshold because pass rate `0.6667` is just below the configured `0.67`. | Borderline threshold case. | This should be described as borderline, not as strong evidence against the model. A strict decimal threshold turns a two-of-three pass into non-stable. |
| `TP73-2` | nonresponder | `2 / 3` label-aware passes; no same-gene nonresponder sibling support; response ratio mean `1.487`, near the `1.5` label-aware threshold. | Borderline nonresponder with weak support. | Do not treat as a clean failure. It is near threshold and lacks same-label sibling support. |

## Stable Pass Context

Stable label-aware passes were observed for:

- `IKZF3-1`
- `MIER1-7`
- `MIER1-8`
- `NFATC1-5`
- `NFATC1-8`
- `SOX5-3`
- `ZNF195-6`
- `ZNF534-1`

The stable pass set includes both responders and nonresponders, but it is weighted toward easier nonresponder cases. Several stable nonresponders pass by the label-aware response-amplitude criterion even when endpoint MSE fraction is above `1.0`, which is appropriate for the current label-aware design but should be described clearly.

## Safe Report Language

The natural isoform benchmark shows partial but incomplete reliability: `8 / 14` holdouts were stable label-aware passes, while the remaining cases expose under-transport of strong responders, unsupported nonresponder holdouts, and seed sensitivity near decision thresholds. This is enough to use the benchmark as a guardrail for mutant interpretation, but not enough to claim robust generalization to unseen TF isoforms.

For mutant transport, the correct framing is:

> The mutant experiment tests whether the frozen sequence-conditioned endpoint model is sensitive to curated sequence perturbations. The result is encouraging for HNF4A, but it should be interpreted alongside the natural isoform benchmark, where the same endpoint model remains imperfect and seed-sensitive.
