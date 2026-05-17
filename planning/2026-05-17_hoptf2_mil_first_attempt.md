# HopTF2 MIL First Attempt

Date: 2026-05-17

## Objective

Train on Joung `H1 human embryonic stem cells`, learn a TF perturbation embedding that remains useful across H1 differentiation contexts, and test whether the same embedding transfers to `SCP3357` `HepG2`.

The HopTF retrieval object is
$$
q_{\mathrm{TF}} = W_q \, \mathrm{ESM\text{-}C}(s_{\mathrm{TF}}),
\qquad
\alpha_{\mathrm{TF}} = \mathrm{softmax}\!\left(\beta K_{\mathrm{AG}} q_{\mathrm{TF}}\right),
\qquad
\mu_{\mathrm{TF}} = V_{\mathrm{AG}}^\top \alpha_{\mathrm{TF}},
$$
with
$$
K_{\mathrm{AG}} \in \mathbb{R}^{N \times d},
\qquad
V_{\mathrm{AG}} \in \mathbb{R}^{N \times d}.
$$

The working hypothesis is that same-gene TF isoforms share a small sequence-side prototype basis in AlphaGenome retrieval space, each isoform is a mixture over that basis, and H1 differentiation state and then H1 to HepG2 mainly change mixture weights over a relatively stable basis.

Support for that hypothesis means:

- single-cell response bags align with the isoform-side basis
- the basis remains predictive on held-out H1 contexts
- the same basis improves transfer to `SCP3357` `HepG2`

## Hypothesis

The progression to test is:

1. **Isoform to single-cell response**
If the basis is real, single-cell responses for one isoform should reflect which prototypes are active.

2. **Isoform to H1 differentiation context**
If the basis is real, moving across H1 contexts should mostly change mixture weights over the same prototypes.

3. **Isoform to HepG2 transfer**
If the basis is real and stable enough, the same prototypes should remain useful when moving from Joung H1 to `SCP3357` `HepG2`.

Compact picture:

```text
same-gene isoforms
        |
        v
shared prototype basis
        |
        +--> 1. single-cell response realization
        |
        +--> 2. H1 context / differentiation shift
        |
        +--> 3. H1 -> HepG2 transfer
```

In compact form:

$$
q_i = W_q \, \mathrm{ESM\text{-}C}(s_i),
\qquad
\alpha_i = \mathrm{softmax}\!\left(\beta K_{\mathrm{AG}} q_i\right),
\qquad
\mu_i = V_{\mathrm{AG}}^\top \alpha_i
$$

for isoform \(i\), together with a same-gene prototype basis

$$
h_g^1, \dots, h_g^K \in \mathbb{R}^d,
\qquad
\pi_i \in \Delta^K,
\qquad
z_i = \sum_{k=1}^K \pi_i^k h_g^k,
$$

Definitions:

$$
\begin{aligned}
g &:= \text{parent TF gene of isoform } i \\
z_i &:= \text{isoform-side perturbation representation} \\
\pi_i &:= \text{isoform-specific mixture weights over shared prototypes}
\end{aligned}
$$

## Experiment Root

Put all new files for this attempt under
`~/MIL/experiments/v1_isoform_basis_transfer/`.

Use this layout:

- `~/MIL/experiments/v1_isoform_basis_transfer/scripts/`
- `~/MIL/experiments/v1_isoform_basis_transfer/inputs/`
- `~/MIL/experiments/v1_isoform_basis_transfer/checkpoints/`
- `~/MIL/experiments/v1_isoform_basis_transfer/logs/`
- `~/MIL/experiments/v1_isoform_basis_transfer/diagnostics/`
- `~/MIL/experiments/v1_isoform_basis_transfer/tests/`
- `~/MIL/experiments/v1_isoform_basis_transfer/JOURNAL.md`

## Current HopTF Path

`~/scripts/train_hopfield_projection.py` takes `ESM-C`, projects it into the AlphaGenome key space, scores it against genome keys, applies a softmax over the genome, and retrieves a value mixture.

That softmax-over-genome mixture is the core sequence-side object in HopTF. This note tests whether it should be factorized into a small number of reusable prototype modes.

## Method Overview

MIL enters through two coupled views of the same latent object.

The sequence-side view uses one bag per TF gene. The bag members are all isoforms from that gene. The bag target is a shared prototype basis in AlphaGenome retrieval space, and each isoform is modeled as a mixture over that basis.

The response-side view uses one bag per perturbation. The bag members are the single-cell response rows for that perturbation. The pooled response representation is then compared against the isoform-side representation.

The transfer question is whether the same prototype basis remains useful across H1 differentiation contexts and then from H1 to HepG2.

## Sequence-Side Prototype Model

The sequence-side model tests whether same-gene isoforms share a reusable prototype basis.

## Response-Side Realization

The response-side model tests whether single-cell responses align with that prototype basis.

The response-side realization is

$$
x_{ic} \in \mathbb{R}^m,
\qquad
r_{ic} = f_{\mathrm{cell}}(x_{ic}),
\qquad
a_{ic} = \mathrm{softmax}_c\!\big(w^\top \tanh(U r_{ic})\big),
$$

$$
\tilde z_i = \sum_c a_{ic} r_{ic},
$$

Definitions:

$$
\begin{aligned}
x_{ic} &:= \text{response row for cell } c \text{ under isoform perturbation } i \\
r_{ic} &:= \text{cell embedding} \\
a_{ic} &:= \text{attention weight for cell } c \\
\tilde z_i &:= \text{bag-level response representation}
\end{aligned}
$$

The alignment target is

$$
\tilde z_i \approx z_i
$$

## Context Shift and Transfer

The context-shift model is

$$
z_{i,\kappa} = \sum_{k=1}^K \pi_{i,\kappa}^k h_g^k,
$$

Definitions:

$$
\begin{aligned}
\kappa &:= \text{cell context, differentiation state, or cell line} \\
z_{i,\kappa} &:= \text{context-specific perturbation representation for isoform } i \\
\pi_{i,\kappa}^k &:= \text{weight of prototype } k \text{ in context } \kappa
\end{aligned}
$$

The transfer hypothesis is

$$
h_g^k \text{ is relatively stable across contexts, while } \pi_{i,\kappa} \text{ changes with } \kappa.
$$

The intended behavior is a stable prototype basis with context-dependent mixture weights across H1 differentiation contexts and then from H1 to HepG2. If that behavior is absent, the H1-to-HepG2 transfer objective is weak.

## First-Pass Design

Use one sequence-side student and up to two teachers.

- sequence input: `ESM-C`
- sequence-side bag: same-gene isoforms
- response-side bag: per-cell PCA rows
- retrieval memory: the existing AlphaGenome keys and values
- endpoint target: `latent_pca`

Use `latent_pca` in the first pass so the experiment changes the MIL supervision path without changing the endpoint target used by the existing HopTF training path.

At fit time, the isoform-bag teacher learns a gene-level prototype basis, the student learns that basis from `ESM-C`, and the cell-bag teacher checks whether those prototypes align with perturbation-level response structure. At tune and hold, the model uses only `ESM-C`.

## Build Plan

Add four files:

1. `~/MIL/experiments/v1_isoform_basis_transfer/scripts/build_isoform_gene_bags.py`
- groups isoforms by `gene_symbol`
- writes `~/MIL/experiments/v1_isoform_basis_transfer/inputs/isoform_gene_bags.npz`
- arrays: `gene_symbol`, `bag_isoforms`, `bag_mask`, `bag_source`

2. `~/MIL/experiments/v1_isoform_basis_transfer/scripts/stage_perturbation_latents.py`
- copies `tmp/hopfield_fitting/perturbation_latents.npz` into the experiment `inputs/` directory
- writes `~/MIL/experiments/v1_isoform_basis_transfer/inputs/perturbation_latents.npz`

3. `~/MIL/experiments/v1_isoform_basis_transfer/scripts/build_perturbation_cell_bags.py`
- groups per-cell PCA rows by `perturbation_id`
- writes `~/MIL/experiments/v1_isoform_basis_transfer/inputs/perturbation_cell_bags.npz`
- arrays: `perturbation_id`, `bag_cells`, `bag_mask`, `bag_source`
- cap each bag at `64` cells

4. `~/MIL/experiments/v1_isoform_basis_transfer/scripts/train_hoptf2_mil.py`
- loads `ESM-C`, AlphaGenome memory, isoform bags, and optional cell bags
- trains the isoform-bag teacher
- retrains the HopTF query projection from `ESM-C`
- optionally aligns the student to the cell-bag teacher
- evaluates on fit, tune, and hold

This is a separate direct endpoint model. It does not patch `scripts/train_otcfm_sequence_conditioned.py`.

## Model Layout

### Teacher A: isoform-bag teacher

Inputs:
- same-gene isoform bag
- `ESM-C`

Core outputs:
- gene-level prototype vectors
- isoform-specific mixture weights

Training target:

$$
z_i = \sum_{k=1}^K \pi_i^k h_g^k
$$

The resulting \(z_i\) should predict the perturbation endpoint while preserving same-gene isoform structure.

### Teacher B: cell-bag teacher

Inputs:
- per-cell PCA bag
- optional prototype context

Role:
- predict `latent_pca`
- test whether cell-level response structure aligns with the sequence-side prototype basis

Response-side object:

$$
\tilde z_i = \sum_c a_{ic} r_{ic},
$$

Alignment question:

$$
\tilde z_i \approx z_i
$$

### Student

Input:
- `ESM-C`

Core path:
- project into AlphaGenome key space
- retrieve a genome mixture
- map the mixture to prototype-aligned latent(s) and an endpoint prediction

Train the student with:

- endpoint loss to `latent_pca`
- latent alignment loss to the isoform-bag teacher
- optional alignment loss to the cell-bag teacher when the response-side branch is enabled

In compact form, the student losses are:

$$
\mathcal{L}_{\mathrm{endpoint}} =
\ell\!\left(\hat y_i, y_i\right),
\qquad
\mathcal{L}_{\mathrm{iso}} =
\left\lVert \hat z_i - z_i \right\rVert_2^2,
\qquad
\mathcal{L}_{\mathrm{cell}} =
\left\lVert \hat z_i - \tilde z_i \right\rVert_2^2,
$$

with \(\mathcal{L}_{\mathrm{cell}}\) enabled only when the response-side alignment branch is active.

Follow-up loss:
- retrieval-geometry loss that matches teacher neighbor structure across fit perturbations

## Multi-Prototype Retrieval

If one-vector retrieval underfits, the next step is a multi-prototype student:

- predict `K` sequence-side query slots from `ESM-C`
- let each slot retrieve its own AlphaGenome mixture
- pool those mixtures into one endpoint prediction

That tests whether one TF isoform should be represented as a mixture over a small number of reusable AlphaGenome modes.

For the first pass, keep `K = 1` in the baseline and treat `K > 1` as the first architectural extension.

## Validation Ladder

Use one short ladder:

1. **Isoform-family probes**
- compute within-gene isoform counts, pairwise `ESM-C` distances, pairwise baseline HopTF-query distances, and simple clustering quality
- summarize genes with at least `2` and at least `3` isoforms
- if within-gene structure is weak, the isoform-bag hypothesis is weak

2. **Isoform-bag MIL**
- compare `hoptf_baseline`, `hoptf_isoform_mil_mean_pool`, and `hoptf_isoform_mil_attention`
- ask whether same-gene isoforms share a useful prototype basis and whether that basis improves retrieval

3. **Single-cell alignment probes**
- compute bag size, within-bag covariance trace, `k`-means inertia for `K = 1, 2, 3`, silhouette for `K = 2`, and GMM BIC for `K = 1, 2, 3`
- summarize all perturbations, responder versus weak-response groups, and top `25` most heterogeneous perturbations
- if multimodality is weak, the response-side alignment branch is weak

4. **Cell-bag alignment**
- if Step 2 shows isoform-side structure, enable the cell-bag teacher
- ask whether single-cell responses align with the isoform-side prototype basis
- read out grouped endpoint metrics, teacher-student latent cosine, prototype usage entropy, and within-bag cluster-to-prototype alignment

5. **Multi-prototype retrieval**
- if Step 2 or Step 4 still underfits, compare one-query HopTF against `K`-prototype HopTF
- read out grouped endpoint metrics, genome-attention entropy, effective number of loci, top-k locus overlap, and per-mode weight shifts

6. **Within-H1 transfer**
- train on one subset of Joung H1 and test on held-out H1 differentiation contexts
- ask whether the prototype basis remains useful across H1 contexts

7. **H1 to HepG2 transfer**
- only if Step 6 works, evaluate transfer to `SCP3357` `HepG2`
- ask whether a basis learned in H1 remains useful in HepG2 and whether H1 and HepG2 mainly differ in mixture weights over that basis

## Failure-Mode Branches

If results are mixed, isolate the sequence, response, and transfer layers.

### Branch A: isoform heterogeneity only

- do same-gene isoforms define a useful prototype basis even without cell-bag supervision
- train the isoform-bag teacher only
- train the student to match that basis
- evaluate same-gene held-out isoform prediction

### Branch B: single-cell heterogeneity only

- do bags show real multi-state structure within one perturbation even without isoform-bag supervision
- train the cell-bag teacher only
- predict perturbation endpoint from bag rows
- compare mean pooling versus attention pooling

### Branch C: retrieval shape only

- does one TF sequence need more than one retrieval mode even without bag supervision
- compare one-query HopTF versus `K`-prototype HopTF
- use sequence-side training only
- evaluate on grouped metrics

### Branch D: transfer only

- do perturbation representations learned in H1 remain useful across held-out H1 contexts or in HepG2
- fix one perturbation representation family
- evaluate within-H1 transfer and H1 to `SCP3357` `HepG2` transfer directly

## Inputs

- `PERTURBATION_METADATA_hard_local_subsample.csv`
  Fields: `perturbation_id`, `isoform_id`, `isoform_embedding_id`, `gene_symbol`, `n_cells`, `response_score`, `protein_aa_length`, `label_status`
- `tf_atlas_morf_isoforms_esmc_600m_mean_non_special.npy`
- `tf_atlas_morf_isoform_vocab.json`
- `~/MIL/experiments/v1_isoform_basis_transfer/inputs/perturbation_latents.npz`
  Fields: `perturbation_id`, `latent_pca`, `control_mean`, `control_sd`
- `~/MIL/experiments/v1_isoform_basis_transfer/inputs/isoform_gene_bags.npz`
- `~/MIL/experiments/v1_isoform_basis_transfer/inputs/perturbation_cell_bags.npz`

On the first pass, `n_cells`, `response_score`, and `protein_aa_length` are for selection and diagnostics, not main model inputs.

## Outputs

- `~/MIL/experiments/v1_isoform_basis_transfer/scripts/build_isoform_gene_bags.py` -> `~/MIL/experiments/v1_isoform_basis_transfer/inputs/isoform_gene_bags.npz`
- `~/MIL/experiments/v1_isoform_basis_transfer/scripts/build_perturbation_cell_bags.py` -> `~/MIL/experiments/v1_isoform_basis_transfer/inputs/perturbation_cell_bags.npz`
- `~/MIL/experiments/v1_isoform_basis_transfer/scripts/stage_perturbation_latents.py` -> `~/MIL/experiments/v1_isoform_basis_transfer/inputs/perturbation_latents.npz`
- `~/MIL/experiments/v1_isoform_basis_transfer/scripts/train_hoptf2_mil.py` -> `~/MIL/experiments/v1_isoform_basis_transfer/checkpoints/` and `~/MIL/experiments/v1_isoform_basis_transfer/diagnostics/`
- `~/MIL/experiments/v1_isoform_basis_transfer/logs/run_v1_isoform_basis_transfer.log`
- `~/MIL/experiments/v1_isoform_basis_transfer/diagnostics/V1_ISOFORM_BASIS_TRANSFER_REPORT.md`

## Comparisons

Core runs:
- `hoptf_baseline`
- `hoptf_isoform_mil_mean_pool`
- `hoptf_isoform_mil_attention`

Follow-up runs:
- `hoptf_isoform_plus_cell_mil`
- `hoptf_proto_k`

## Metrics

Primary table columns:

- grouped `mse`
- grouped `mean_cosine_similarity`
- mean holdout `control_standardized_endpoint_mse_fraction_of_baseline`
- panel `n_label_aware_passed`
- panel `n_stable_label_aware_passed`

Secondary diagnostics:

- isoform-family structure probe summaries
- single-cell heterogeneity probe summaries
- teacher fit and tune endpoint MSE
- student fit and tune endpoint MSE
- teacher-student latent cosine
- mean genome-attention entropy
- mean holdout `mean_control_standardized_response_l2_ratio`
- within-family prototype concentration
- within-bag cluster to prototype alignment

## Implementation Order

1. Build same-gene isoform bags.
2. Stage `perturbation_latents.npz` into the experiment `inputs/` directory.
3. Run isoform-family probes.
4. Train the isoform-bag teacher and sequence-side student.
5. Compare `hoptf_baseline`, `hoptf_isoform_mil_mean_pool`, and `hoptf_isoform_mil_attention`.
6. Build per-cell PCA bags.
7. Run single-cell heterogeneity probes.
8. If Step 4 helps, enable the cell-bag alignment branch.
9. If the model still underfits, test `K > 1` retrieval.
10. If the candidate helps, run within-H1 context transfer.
11. If within-H1 transfer helps, run H1 to `SCP3357` `HepG2` transfer.

## Summary

Same-gene isoforms define the main MIL object. The model learns a prototype basis in AlphaGenome retrieval space and maps each isoform to mixture weights over that basis. Single-cell bags are used to test whether those prototypes appear in real response structure. Transfer tests ask whether the same basis stays useful across H1 contexts and then from H1 to HepG2.
