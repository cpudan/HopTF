# Linear Probe Results

This report contains the two main label surfaces:

1. hard responder labels from the original quartile split
2. soft responder labels from a median `response_score` split

Both runs use the full TF set. There is no Borzoi feature and no chromosome coverage restriction.

## Common Setup

- grouping: `gene_symbol`
- cross-validation: `StratifiedGroupKFold`, `5` folds
- embedding family: `ESM-C 600M`
- feature sets:
  - `esmc`
  - `protein_length`
  - `aa_composition`
  - `gene_symbol_onehot`
  - `esmc` with shuffled labels

## Hard Labels

- labeled rows: `1712`
- positives: `856`
- negatives: `856`
- source: original responder/nonresponder quartile labeling

| Feature set | Label source | AUROC | AUPRC | Balanced accuracy |
| :--- | :--- | ---: | ---: | ---: |
| `protein_length` | metadata | 0.6533 | 0.6250 | 0.6139 |
| `esmc` | metadata | 0.6185 | 0.6113 | 0.5940 |
| `aa_composition` | metadata | 0.6038 | 0.5915 | 0.5754 |
| `esmc` | shuffled labels | 0.5117 | 0.5135 | 0.5035 |
| `gene_symbol_onehot` | metadata | 0.4862 | 0.4906 | 0.4907 |

Read:
- `ESM-C` is clearly above the shuffled-label null.
- `ESM-C` beats amino-acid composition.
- `gene_symbol_onehot` stays at chance under grouped CV.
- `protein_length` is still the strongest simple baseline on this hard-label surface.

## Soft 0.5 Labels

- labeled rows: `3421`
- positives: `1711`
- negatives: `1710`
- threshold: `response_score >= 2.624636650085449`
- source: median split over all eligible non-control TF rows

| Feature set | Label source | AUROC | AUPRC | Balanced accuracy |
| :--- | :--- | ---: | ---: | ---: |
| `esmc` | metadata | 0.5889 | 0.5816 | 0.5683 |
| `protein_length` | metadata | 0.5688 | 0.5159 | 0.5820 |
| `aa_composition` | metadata | 0.5672 | 0.5492 | 0.5533 |
| `esmc` | shuffled labels | 0.4833 | 0.4890 | 0.4917 |
| `gene_symbol_onehot` | metadata | 0.4803 | 0.4886 | 0.4832 |

Read:
- `ESM-C` is the strongest feature set on AUROC and AUPRC under the median split.
- `protein_length` still carries signal, but it is weaker than `ESM-C` on ranking metrics.
- `gene_symbol_onehot` stays at chance under grouped CV.
- the shuffled-label control stays near chance.


## Hard Labels: Length-Binned Control

The hard-label surface was also checked within protein-length quartiles so `ESM-C` was not only benefiting from mixing short and long proteins.

| Quartile | Mean length | AUROC | AUPRC | Balanced accuracy | Rows |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Q1` | 363.9 | 0.5837 | 0.7485 | 0.5571 | 429 |
| `Q2` | 525.1 | 0.5866 | 0.6793 | 0.5729 | 427 |
| `Q3` | 677.1 | 0.5924 | 0.5060 | 0.5792 | 429 |
| `Q4` | 1104.4 | 0.5849 | 0.4335 | 0.5556 | 427 |

Read:
- `ESM-C` stays above chance by AUROC in every length quartile.
- the signal is not only coming from mixing short and long proteins together.
- AUPRC shifts across quartiles because responder prevalence changes across the bins.

## Files

- hard-label summary: `hoptf/colleague_scripts/outputs/linear_probe_report/SUMMARY.json`
- soft-label summary: `hoptf/colleague_scripts/outputs/linear_probe_report/SUMMARY_soft.json`
- soft-label threshold summary: `hoptf/colleague_scripts/outputs/linear_probe_report/SOFT_LABEL_SUMMARY.json`
