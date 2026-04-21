# Linear Probe Report

## Scope

This report collects the current standalone probe results for:

- the primary hard-label responder probe
- the matched Borzoi panel subset
- the softer median-split probe
- length-control checks on `ESM-C`

## 1. Primary Hard-Label Probe

Label surface:

- keep only `responder` and `nonresponder`
- drop the ambiguous middle

Rows:

- `1712` labeled perturbations

Results:

| Feature set | AUROC | AUPRC | Balanced accuracy |
| :--- | ---: | ---: | ---: |
| `esmc` | 0.6210 | 0.6149 | 0.5929 |
| `protein_length` | 0.6533 | 0.6250 | 0.6139 |
| `aa_composition` | 0.6038 | 0.5915 | 0.5754 |
| `gene_symbol_onehot` | 0.4862 | 0.4906 | 0.4907 |
| `esmc_shuffled_labels` | 0.5117 | 0.5135 | 0.5035 |

Read:

- `ESM-C` carries real signal above the shuffled null.
- `protein_length` is a strong nuisance baseline.
- `gene_symbol_onehot` stays useless under grouped CV.


## 2. Soft Label Probe

Soft label definition:

- `responder_score_soft = 1` if `response_score >= median(response_score)`
- `responder_score_soft = 0` otherwise

Rows:

- `3421`

Median threshold:

- `2.624636650085449`

Results:

| Feature set | AUROC | AUPRC | Balanced accuracy |
| :--- | ---: | ---: | ---: |
| `esmc` | 0.5894 | 0.5814 | 0.5665 |
| `protein_length` | 0.5688 | 0.5159 | 0.5820 |
| `aa_composition` | 0.5672 | 0.5492 | 0.5533 |
| `gene_symbol_onehot` | 0.4803 | 0.4886 | 0.4832 |
| `esmc_shuffled_labels` | 0.4827 | 0.4874 | 0.4887 |

Read:

- on the softer median-split label surface, `ESM-C` is stronger than `protein_length` by AUROC and AUPRC
- this is more consistent with the concern that hard quartile labels over-amplify nuisance structure

## 3. Length Controls

These checks ask whether `ESM-C` still carries signal after controlling for protein length.

### 3.1 Residualized `ESM-C`

Procedure:

- within each CV fold, regress each embedding dimension on `protein_aa_length` using the training split
- subtract the predicted length component from both train and test embeddings
- run the same grouped logistic probe on the residualized features

Results on the hard-label surface:

| Feature set | AUROC | AUPRC | Balanced accuracy |
| :--- | ---: | ---: | ---: |
| `esmc` | 0.6210 | 0.6149 | 0.5929 |
| `esmc_length_residualized` | 0.5985 | 0.5926 | 0.5718 |

Read:

- performance drops after residualizing out length
- the drop is real but not catastrophic
- `ESM-C` retains signal beyond a simple linear length component

### 3.2 `ESM-C` Within Length Quartiles

Hard-label surface, grouped CV, evaluated separately within length quartiles:

| Quartile | Mean length | AUROC | AUPRC | Balanced accuracy | Rows |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Q1 | 363.9 | 0.5837 | 0.7485 | 0.5571 | 429 |
| Q2 | 525.1 | 0.5866 | 0.6793 | 0.5729 | 427 |
| Q3 | 677.1 | 0.5924 | 0.5060 | 0.5792 | 429 |
| Q4 | 1104.4 | 0.5849 | 0.4335 | 0.5556 | 427 |

Read:

- `ESM-C` remains above chance within every length quartile
- signal is not coming only from mixing short and long proteins together
- AUPRC changes strongly across bins because class balance changes across the quartiles

## 4. Bottom Line

- `ESM-C` is a real signal-bearing TF embedding on this responder probe
- `protein_length` is a strong nuisance baseline on the hard quartile labels
- the softer median-split probe weakens the length story and improves the relative standing of `ESM-C`
- residualizing length hurts `ESM-C`, but it does not erase the signal
- within-bin probes show that `ESM-C` is not purely a length detector
