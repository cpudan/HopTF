# HopTF Benign-Variant MIL Experiment Plan

Date: 2026-05-18

## Executive Summary

This experiment tests whether HopTF can be trained to be stable to tolerated TF
sequence variation without losing sensitivity to damaging TF sequence
perturbations.

The current HopTF evidence is mixed in a way that makes this worth doing but also
easy to overclaim. The SCARF-space controlled sequence benchmark does not show
that current sequence or Hopfield-query features improve held-out perturbed-state
prediction after protein length, ORF length, and recovered cell count are
included. The stronger positive result is the matched variant diagnostic:
pathogenic missense variants produce larger absolute predicted response changes
than matched random substitutions and matched benign/control variants, while the
signed weakening result is not consistent. Retrieval movement also looks stronger
against matched random controls than against older matched benign controls.

The proposed MIL experiment turns the regenerated benign/control panel into
training supervision instead of using it only as a post hoc control. For each TF
isoform, create a bag containing the wild-type sequence and up to 10 tolerated or
benign/control sequence instances. All instances in that bag share the same TF
Atlas response label, because the working assumption is that tolerated variants
should preserve the isoform-level perturbation response. The model should learn a
response and retrieval representation that is stable across those tolerated
instances.

The key success criterion is a three-part result:

1. MIL training improves or at least does not materially worsen held-out response
   prediction relative to the current reference-only HopTF setup.
2. MIL training reduces within-bag dispersion among wild-type and tolerated
   sequence instances in query, retrieval, retrieved representation, and
   predicted response space.
3. Pathogenic variants that currently evoke strong absolute response or retrieval
   shifts still remain out-of-distribution relative to the tolerated bag after MIL
   training.

If all three hold, the story is clean: HopTF can learn tolerance-aware invariance
without collapsing sequence sensitivity. If only the stability part holds, this
is a useful regularization diagnostic but not a main positive result. If strong
pathogenic variants become in-distribution after training, the experiment fails
for the paper even if endpoint MSE improves.

## Why This Fits The HopTF Story

The paper currently frames HopTF as a sequence-conditioned associative retrieval
model with a weakly supervised multiple-instance structure: the model observes a
TF perturbation response but not the causal genomic loci or exact sequence
features that mediate it. The existing draft already identifies the planned MIL
extension as the strongest next training experiment.

This experiment fits the story because it tests the most important distinction
for a biologically useful sequence-conditioned model:

- tolerated sequence variation should map near the wild-type response program;
- damaging sequence variation should still perturb retrieval or predicted
  response;
- the retrieval distribution should remain auditable rather than becoming an
  opaque average over all sequence inputs.

The experiment should be written as "tolerance-aware sequence regularization for
HopTF", not as a clinical pathogenicity classifier. The held-out pathogenic
variants are a stress test of whether the learned invariance is selective.

## Literature Grounding

The MIL literature supports this experiment, but only under a careful
formulation.

- Classic MIL was introduced for settings where a bag has many possible
  instances but only a bag-level label is observed, such as molecule
  conformations with a drug-activity label. See Dietterich, Lathrop, and
  Lozano-Perez, 1997:
  https://doi.org/10.1016/S0004-3702(96)00034-3

- The standard MIL assumption, where a positive bag is positive because at least
  one instance is positive, is not the right assumption for this HopTF design.
  Reviews emphasize that MIL methods depend strongly on the assumed relation
  between instances and bag labels. See Foulds and Frank, 2010, and Carbonneau
  et al., 2018:
  https://www.cambridge.org/core/journals/knowledge-engineering-review/article/review-of-multiinstance-learning-assumptions/0915098C83BF119A377015A45952247A
  https://arxiv.org/abs/1612.03365

- This experiment is closer to multiple-instance regression and collective/set
  learning: each TF isoform bag has a real-valued response vector, and the bag
  label is assumed to reflect the tolerated sequence family, not a single hidden
  positive witness. Ray and Page explicitly formulated multiple-instance
  regression with real-valued bag labels:
  https://pages.cs.wisc.edu/~sray/papers/mip.reg.icml01.pdf

- Because the tolerated variants inside one bag are unordered, the model should
  use permutation-invariant pooling or attention over instances. Deep Sets gives
  the basic set-function justification, while attention-based MIL and Set
  Transformer motivate learnable attention pooling over unordered bags:
  https://papers.nips.cc/paper/6931-deep-sets
  https://proceedings.mlr.press/v80/ilse18a.html
  https://proceedings.mlr.press/v97/lee19d.html

- The Hopfield/attention connection is especially relevant for HopTF. DeepRC used
  modern Hopfield networks / transformer-like attention for a biological MIL
  problem with massive bags and sparse witnesses. The analogy is useful, but
  HopTF differs because its bags are tolerated protein variants for one TF
  isoform and the target is a perturbation-response vector:
  https://arxiv.org/abs/2007.13505

Design implication: do not call any attention pooling model a success by default.
Compare against a reference-only model, an instance-replication baseline, and a
mean-pooling bag baseline. Report whether attention weights are stable and
whether the model preserves pathogenic OOD behavior.

## Core Question

Can training on up to 10 tolerated or benign/control variants per TF isoform make
HopTF's sequence-conditioned retrieval and predicted response more stable while
preserving the out-of-distribution response of strong pathogenic variants?

Operational subquestions:

- Does MIL training improve held-out-gene SCARF response prediction over the
  current reference-only setup or at least avoid worsening it?
- Does MIL training reduce the spread of predicted response and retrieval metrics
  across wild-type plus benign/control instances within the same isoform bag?
- Are strong pathogenic variants still outside the tolerated bag distribution
  after MIL training?
- Does the number of benign/control variants per bag matter?
- Do higher-quality benign/control tiers outperform lower-quality or fallback
  controls?

## One-Pass Execution Contract

This experiment should be runnable in one pass only if the preflight checks pass.
Do not start model training until the following are locked and written into a run
manifest:

- source paths, file sizes, and checksums for SCARF response artifacts;
- source paths, file sizes, and checksums for benign/control variant tables;
- source paths, file sizes, and checksums for pathogenic and matched-control
  variant panels;
- ESM-C embedding matrix paths, vocab paths, embedding dimensions, and row counts;
- AlphaGenome memory paths, dimensions, and genome-window metadata path;
- gene-level split assignments;
- model configs, loss weights, bag-size policy, tier policy, and random seeds.

Hard preflight gates:

- SCARF response metadata must join to perturbation metadata by
  `isoform_embedding_id` for at least 95 percent of non-control isoforms used in
  training.
- Benign/control bag rows must sequence-round-trip against the local HopTF
  wild-type sequence. Reference amino acid and one-based position must match.
- Every training bag must have exactly one wild-type instance and zero or more
  eligible tolerated/control instances after tier filtering.
- At least 1,000 isoforms must have one or more Tier 1-4 tolerated/control
  instances, or the primary MIL claim should be downgraded to a feasibility
  smoke test.
- The locked strong pathogenic OOD panel must contain at least 10 variants with
  exact isoform mapping and an eligible same-isoform tolerated/control bag. If
  fewer than 10 pass, do not use the OOD-retention result as a pass/fail gate;
  report it as qualitative.
- The reference-only baseline must be rerun or re-scored on the exact same split
  assignments used by MIL.

Hard stop conditions:

- any source table lacks stable row IDs needed to join sequence, embedding,
  response, and tier metadata;
- any split has the same `gene_symbol` in fit and tune/hold;
- pathogenic variants or matched random controls enter model fitting or
  hyperparameter selection;
- Tier 5 fallback rows are mixed into the primary Tier 1-4 run;
- the selected model condition is chosen using pathogenic OOD performance.

## Data Inputs

### Required HopTF Artifacts

- Perturbation metadata:
  `data/processed/linear_probe/tfatlas_subsample/PERTURBATION_METADATA_hard_local_subsample.csv`
- SCARF response artifacts from the May 17 response-representation rerun, or the
  equivalent artifact to be staged before this experiment:
  - `scarf_response_latents.npz`
  - `scarf_response_metadata.csv`
  - `scarf_response_report.json`
- Current HopTF sequence inputs:
  - `tf_atlas_morf_isoforms_esmc_600m_mean_non_special.npy`
  - `tf_atlas_morf_isoform_vocab.json`
- AlphaGenome key/value memory artifacts used by the current HopTF retrieval path.
- Current reference-only HopTF or Hopfield-query baseline outputs used for the
  SCARF-space controlled benchmark.

### Benign/Control Variant Bags

Use the regenerated fixed-size control table described in the draft:

```text
data/processed/variant_controls/tf_isoform_benign_controls_10_per_isoform_ortholog_tier4_20260517/
hf://datasets/pvd232/HopTF/processed/variant_controls/tf_isoform_benign_controls_10_per_isoform_ortholog_tier4_20260517
```

The local directory was referenced by the draft, but it was not present in this
workspace during this planning pass. The implementation should first stage or
verify the frozen files from the shared cluster or Hugging Face mirror.

Expected table properties from the current draft:

- 32,640 total capped rows.
- 10 control rows for each of 3,264 non-control TF isoforms.
- Natural evidence selected first: ClinVar benign/likely benign, gnomAD
  population-tolerated rows, and MaveDB functional-control rows.
- Tier 4 ortholog-supported synthetic controls fill many remaining slots.
- Tier 5 conservative fallback controls fill only the remaining slots needed to
  keep 10 rows per isoform.

Primary MIL training should use Tiers 1-4. Tier 5 rows should be excluded from
the primary claim and used only in an explicit fallback ablation.

### Pathogenic Held-Out Variant Panel

Use the locked pathogenic and matched-control variant panels from the May 17
SCARF-space matched variant analysis wherever possible. If the full files are not
local, regenerate the same locked panel before model training begins.

Pathogenic variants are never used for model selection in the primary analysis.
They are a final held-out stress test.

For the OOD-retention test, keep only pathogenic variants whose wild-type isoform
has a usable tolerated/control bag under the evaluated tier policy. Pathogenic
variants without same-isoform tolerated controls can still be used for
reference-only variant summaries, but they cannot define a tolerated-bag OOD
score.

## Bag Definition

For each TF isoform `i`, define a sequence bag:

```text
B_i = {wild_type_i, tolerated_i1, tolerated_i2, ..., tolerated_iK}
```

where `K` is varied by experiment and is at most 10 in the fixed-size run.

Each instance in the bag has:

- `isoform_embedding_id`
- `gene_symbol`
- `sequence_instance_id`
- `instance_type`: `wild_type`, `tier1_clinvar_benign`, `tier2_gnomad_strong`,
  `tier2_gnomad_moderate`, `tier3_functional`, `tier4_ortholog_supported`, or
  `tier5_fallback`
- `variant_source`
- `evidence_tier`
- `hgvs_p`
- `one_letter_mutation`
- `protein_position`
- `wt_aa`
- `mut_aa`
- `mutant_sequence`
- `esm_c_embedding_id`
- `esm_c_distance_from_wt`
- quality and exclusion flags from the benign/control curation pipeline.

The bag-level target is the isoform's observed perturbation-response vector:

```text
y_i = SCARF mean(TF-perturbed cells for isoform i) - SCARF mean(control cells)
```

Use the older PCA-derived target only as a sensitivity analysis. The main result
should use SCARF because the current paper has shifted to SCARF-space response
diagnostics.

## Why These Bags Are Plausible

This is not classic positive-witness MIL. The assumption is:

> A high-quality tolerated/control variant should preserve the TF isoform's
> perturbation response closely enough that it can share the wild-type bag-level
> response label.

That assumption is strongest for Tier 1-3 natural or functional evidence, weaker
for Tier 4 ortholog-supported controls, and weakest for Tier 5 fallback rows.
Therefore, quality tiers must be part of the experimental design rather than
metadata hidden until the end.

## Experimental Conditions

Run a small but complete model grid. The first pass should prioritize clean
comparisons over architectural novelty.

### Condition 0: Reference-Only Current Setup

Name:

```text
hoptf_reference_only
```

Description:

- Train or evaluate using only the canonical wild-type TF isoform sequence.
- This is the current setup.
- Use the same train/tune/hold splits as the MIL runs.

Role:

- Primary comparator for held-out response prediction.
- Comparator for benign stability and pathogenic OOD behavior.

### Condition 1: Instance-Replication Baseline

Name:

```text
hoptf_benign_instance_replicate
```

Description:

- Treat every tolerated/control sequence instance as an independent training row
  with the same response target as the wild-type isoform.
- Do not use bag pooling.

Role:

- Tests whether a naive label-duplication strategy already gives the apparent
  benefit.
- If this performs as well as MIL, the paper should not claim a specific MIL
  advantage.

### Condition 2: Mean-Pooled MIL

Name:

```text
hoptf_benign_mil_mean_pool
```

Description:

- Compute the HopTF query, retrieval distribution, retrieved representation, and
  predicted response for each sequence instance.
- Average instance-level representations or predictions across the bag using
  permutation-invariant mean pooling.
- Apply the endpoint loss to the pooled prediction.

Role:

- Clean set baseline.
- Stable and hard to overinterpret.

### Condition 3: Attention-Pooled MIL

Name:

```text
hoptf_benign_mil_attention_pool
```

Description:

- Use a learned attention operator over sequence instances in each bag.
- The attention-pooled bag representation predicts the response vector.
- Record attention weights by instance and evidence tier.

Role:

- Tests whether the model benefits from weighting high-quality or more
  informative tolerated instances.
- Must be checked for attention collapse onto wild type or low-quality fallback
  rows.

### Condition 4: Invariance-Regularized HopTF

Name:

```text
hoptf_benign_mil_invariance
```

Description:

- Keep per-instance HopTF predictions.
- Add explicit stability losses that pull tolerated/control instances toward the
  wild-type instance in query, retrieval, retrieved representation, and predicted
  response space.
- Apply these losses only to tolerated/control instances, with quality-tier
  weights.

Role:

- Directly tests the desired biological behavior: tolerated variation should not
  move the response much.
- Highest-priority model if the implementation budget is tight.

### Condition 5: Quality-Weighted Attention MIL

Name:

```text
hoptf_benign_mil_quality_attention
```

Description:

- Same as attention-pooled MIL, but initialize or regularize instance weights
  using evidence-tier quality.
- Tier 1-3 rows get the strongest trust, Tier 4 lower trust, Tier 5 excluded from
  primary or given near-zero weight in fallback-only runs.

Role:

- Tests the user's question about whether higher-quality benign annotations do
  better than lower-quality controls.
- Should not be the first model selected unless simpler MIL baselines work.

## Losses

For sequence instance `j` in isoform bag `B_i`, let:

```text
q_ij       = projected ESM-C query
a_ij       = HopTF attention / retrieval distribution over AlphaGenome windows
mu_ij      = retrieved regulatory-context representation
yhat_ij    = predicted SCARF response vector
y_i        = observed isoform-level SCARF response vector
yhat_bag_i = pooled bag prediction
```

### Endpoint Loss

Use the same response loss as the SCARF benchmark:

```text
L_endpoint = loss(yhat_bag_i, y_i)
```

For instance-replication and invariance-regularized runs, also report:

```text
mean_j loss(yhat_ij, y_i)
```

The endpoint loss should be evaluated in grouped held-out-gene splits.

### Benign Stability Loss

For tolerated/control instances only:

```text
L_stability =
  w_q       * mean_j ||q_ij - q_i_wt||_2^2
+ w_mu      * mean_j ||mu_ij - mu_i_wt||_2^2
+ w_y       * mean_j ||yhat_ij - yhat_i_wt||_2^2
+ w_attn    * mean_j JSD(a_ij, a_i_wt)
```

Use robust losses or clipping for high-ESM-C-distance stress-test controls so a
single questionable control does not dominate training.

Recommended first pass:

- `w_y`: on.
- `w_mu`: on.
- `w_attn`: on only if full or top-k attention distributions are tractable.
- `w_q`: low weight, because some benign variants can move ESM-C space while
  preserving downstream response.

### Tier Weights

Use tier weights only for stability and pooling, not for changing the observed
response target.

Recommended initial weights:

```text
Tier 1 clinical benign/control:          1.00
Tier 2 gnomAD strong population:         0.90
Tier 2 gnomAD moderate population:       0.75
Tier 3 functional-assay control:         1.00
Tier 4 ortholog-supported synthetic:     0.50
Tier 5 conservative fallback:            0.00 primary; 0.25 fallback ablation
```

These weights are design choices and must be reported. Also run an unweighted
Tier 1-4 sensitivity analysis to make sure the result is not an artifact of the
chosen weights.

### Collapse Guard

The model must not become invariant to all sequence changes. Do not train on
pathogenic labels in the primary run, but enforce the guard in model selection:

- Select among MIL hyperparameters using held-out response prediction and benign
  stability only.
- After selection, evaluate pathogenic OOD retention once.
- If all candidate settings that improve benign stability erase pathogenic OOD
  behavior, report the experiment as failed rather than tuning on pathogenic
  variants until it passes.

## Bag Size Sweep

Run the same model family across bag sizes:

```text
K = 0, 1, 2, 4, 6, 8, 10
```

where `K = 0` means wild-type only and `K = 10` means all available fixed-size
control rows for the isoform. After tier filtering, some isoforms will have fewer
than `K` eligible controls. Do not backfill those bags with lower-quality tiers
unless the run is explicitly labeled fallback-inclusive. Instead, use all
eligible controls for that isoform, record `requested_K`, `observed_K`, and
`eligible_tier_count`, and stratify metrics by observed bag size.

For each `K > 0`, construct two subset policies:

1. **Quality-prefix policy**
   - Sort controls by evidence tier and internal quality.
   - Select the best `K` rows.
   - This tests whether progressively adding evidence-backed variants helps.

2. **Random-subset policy**
   - Sample `K` controls from eligible Tier 1-4 rows.
   - Use at least 10 subset seeds for `K = 1, 2, 4` and at least 5 seeds for
     larger `K`.
   - This estimates how sensitive the result is to the particular variants
     chosen.

If the quality-prefix curve improves but random subsets are noisy, the story is:
variant quality matters. If both improve smoothly, the story is: additional
tolerated sequence support regularizes the model. If performance worsens after
adding Tier 4 or Tier 5 rows, do not claim that "10 variants" is inherently best.

## Quality-Tier Sweep

Run at least these training sets:

```text
WT only
WT + Tier 1 only
WT + Tier 1-2 strong
WT + Tier 1-2 all
WT + Tier 1-3 natural/functional
WT + Tier 1-4 natural plus ortholog-supported
WT + Tier 1-5 fixed 10 fallback-inclusive
```

Primary claim should use the best prespecified non-pathogenic model selected
from Tier 1-4. Tier 5 fallback-inclusive runs are engineering diagnostics only.

Quality analysis:

- Compare held-out response error by tier set.
- Compare benign within-bag dispersion by tier set.
- After non-pathogenic model selection is complete, report pathogenic OOD
  retention by prespecified tier set as a diagnostic. Do not use pathogenic OOD
  to choose the winning tier set.
- Test monotonic association between natural-evidence fraction and stability
  improvement using Spearman correlation across run summaries.

If the high-quality smaller bags outperform the full 10-row bags, report that
annotation quality matters more than bag size.

## Splitting And Leakage Control

Use grouped splits by `gene_symbol`, with all isoforms and all benign/control
instances from the same gene assigned to the same split.

Required split rules:

- No sequence instance from a held-out isoform may appear in training.
- No benign/control variant for a held-out gene may appear in training.
- Pathogenic held-out variants are never used for fitting or hyperparameter
  selection.
- Matched random controls for pathogenic variants are not used for fitting in
  the primary analysis.
- If multiple isoforms from the same gene appear in the panel, keep them in the
  same split to avoid learning gene-specific response shortcuts.

Recommended split layout:

- Fit: 70 percent of genes.
- Tune: 15 percent of genes.
- Hold: 15 percent of genes.
- Repeat over 5 split seeds if runtime permits.

For final figures, use the same split seed as the current SCARF benchmark if
there is already a locked split. If no split is locked, lock one before training.

## Primary Metrics

### 1. Held-Out Response Prediction

Report on held-out genes:

- SCARF response vector MSE.
- Median row MSE.
- Mean cosine similarity.
- Control-standardized response MSE if control SD is available.
- Percent MSE change relative to `hoptf_reference_only`.
- Percent MSE change relative to protein length + ORF length + recovered cell
  count baseline.

Primary statistical comparison:

- Paired bootstrap over held-out genes for MSE difference between MIL and
  reference-only HopTF.

Interpretation:

- Strong improvement: MIL improves held-out-gene MSE over reference-only and the
  confidence interval supports improvement.
- Acceptable stability-only improvement: MIL MSE is within 1-2 percent of
  reference-only while substantially improving benign stability.
- Failure: MIL worsens response prediction by more than 2 percent unless the
  paper explicitly frames the run as a retrieval-stability diagnostic only.

### 2. Benign/Tolerated Bag Stability

For each isoform bag, compute dispersion among wild type and tolerated instances:

- `q_l2_to_wt`
- attention Jensen-Shannon divergence to WT
- top-100 and top-500 retrieval overlap with WT
- `mu_l2_to_wt`
- `mu_cosine_to_wt`
- predicted response L2 to WT
- predicted response cosine to WT
- absolute predicted response magnitude change to WT

Bag summary metrics:

- median tolerated-instance dispersion per isoform.
- 90th percentile tolerated-instance dispersion per isoform.
- fraction of tolerated instances within the WT empirical neighborhood.
- tier-stratified dispersion.

Primary statistical comparison:

- Paired bootstrap over isoforms comparing benign dispersion before and after MIL
  training.

Success:

- At least 20 percent reduction in median tolerated-instance predicted-response
  dispersion and retrieved-representation dispersion, without endpoint MSE
  degradation beyond the tolerance above.

### 3. Pathogenic OOD Retention

Define an OOD score for a held-out variant `v` relative to its isoform's tolerated
bag:

```text
OOD_endpoint(v) = robust_z(abs_response_change_v; tolerated bag abs changes)
OOD_retrieval(v) = robust_z(attention_JSD_v; tolerated bag attention JSD)
OOD_mu(v) = robust_z(mu_l2_v; tolerated bag mu_l2)
OOD_composite(v) = mean of rank-normalized OOD_endpoint, OOD_retrieval, OOD_mu
```

Use empirical percentiles when bags have enough tolerated instances:

```text
percentile_v = fraction of tolerated instances with metric <= variant metric
```

For fixed 10-row bags, robust z-scores are coarse. Therefore report both robust
z-scores and empirical ranks, and use bootstrap over genes rather than pretending
that 10 controls per isoform provide a precise distribution.

Primary OOD tests:

- Pathogenic variants have higher OOD composite scores than held-out
  benign/control instances.
- Pathogenic variants have higher OOD composite scores than matched random
  substitutions if random controls are available.
- Strong pathogenic variants locked before MIL remain above the 90th percentile
  of tolerated/control instances or above a robust z-score threshold.

OOD retention criterion:

- Let `S_strong` be the locked set of current strong-response pathogenic
  variants.
- At least 80 percent of `S_strong` should remain OOD after MIL training.
- The median OOD score for `S_strong` should not fall by more than 25 percent
  relative to the reference-only baseline.
- Rank correlation of pathogenic absolute effect between reference-only and MIL
  should be at least 0.6 for the full pathogenic panel and at least directionally
  positive for `S_strong`.

If these fail, the model has learned excessive invariance and should not be
promoted.

OOD missingness rule:

- If a pathogenic variant's isoform has fewer than 4 eligible tolerated/control
  instances under the evaluated tier policy, report its OOD score but exclude it
  from the primary same-isoform OOD pass/fail count.
- If fewer than 10 strong pathogenic variants remain after this filter, report
  OOD retention descriptively and do not use it as the decisive success gate.
- Never pool tolerated instances across genes to rescue the primary OOD test.
  Pooled TF-family or domain-matched tolerated controls can be shown only as a
  sensitivity analysis.

## Locking The Strong-Response Pathogenic Set

Before any MIL training, lock the strong pathogenic set using the current
reference-only model.

Recommended definition:

```text
S_response_strong:
  pathogenic variants in the top 10 percent by absolute predicted SCARF response
  change, with exact isoform mapping and matched controls.

S_retrieval_strong:
  pathogenic variants in the top 10 percent by attention JSD or retrieved-vector
  L2 change from WT.

S_strong:
  union of S_response_strong and S_retrieval_strong, capped to a reviewable set
  of 20-30 variants if needed, with deterministic tie-breaking by gene, variant
  ID, and effect size.
```

Record:

- wild-type isoform ID;
- whether the isoform has at least 4 Tier 1-4 tolerated/control instances;
- variant ID and mutation;
- clinical label and review status;
- baseline absolute response change;
- baseline retrieval JSD;
- matched benign/control and matched random availability;
- ESM-C distance from WT;
- domain class if available.

The locked table should be saved before training:

```text
strong_pathogenic_ood_panel_locked.csv
```

## Confirmation That MIL Does Better Than Current Setup

Do not use a single cherry-picked metric. Use a prespecified decision table.

### Strong Win

Call the MIL experiment a strong positive result if:

- held-out SCARF response MSE improves over `hoptf_reference_only`, or is within
  1 percent while mean cosine improves;
- benign/tolerated dispersion in predicted response and retrieved representation
  decreases by at least 20 percent;
- strong pathogenic variants remain OOD by the retention criteria above;
- the result is stronger for Tier 1-4 than for Tier 1-5 fallback-inclusive runs;
- instance-replication does not explain the full gain.

Paper wording:

> Training on tolerated sequence-instance bags stabilized HopTF retrieval and
> predicted response across benign/control variants while preserving
> out-of-distribution behavior for strong pathogenic substitutions.

### Stability-Only Win

Call it a limited positive result if:

- held-out MSE is not better but is within 1-2 percent of reference-only;
- benign/tolerated dispersion improves by at least 20 percent;
- pathogenic OOD retention passes.

Paper wording:

> Benign-variant MIL acted as a tolerance-aware regularizer rather than a
> predictor-performance improvement.

This can support a planned-extension figure or appendix, not the main predictive
claim.

### Failed Or Unsafe Result

Call it failed or unsafe if:

- held-out MSE worsens by more than 2 percent;
- benign stability improves only by using Tier 5 fallback rows;
- attention pooling collapses onto wild type and ignores the benign instances;
- strong pathogenic variants become in-distribution;
- pathogenic OOD retention requires tuning on pathogenic labels.

Paper wording:

> The tolerated-instance objective over-regularized sequence sensitivity or did
> not improve the current HopTF setup.

## Required Outputs

Root output directory:

```text
tmp/hoptf_benign_variant_mil_20260518/
```

Subdirectories:

```text
inputs/
configs/
checkpoints/
metrics/
figures/
reports/
logs/
```

Required input/staging artifacts:

- `inputs/benign_variant_bags_all.csv`
- `inputs/benign_variant_bags_tier1_4.csv`
- `inputs/benign_variant_bags_tier1_5.csv`
- `inputs/benign_variant_bag_membership.npz`
- `inputs/benign_variant_sequences.fasta`
- `inputs/benign_variant_esmc_vocab.json`
- `inputs/benign_variant_esmc_embeddings.npy`
- `inputs/strong_pathogenic_ood_panel_locked.csv`
- `inputs/pathogenic_holdout_panel.csv`
- `inputs/matched_random_controls.csv`
- `inputs/split_assignments_by_gene.csv`

Required run summaries:

- `metrics/model_comparison_by_split.csv`
- `metrics/heldout_response_prediction_summary.csv`
- `metrics/benign_bag_stability_summary.csv`
- `metrics/benign_bag_stability_by_tier.csv`
- `metrics/pathogenic_ood_retention_summary.csv`
- `metrics/strong_pathogenic_ood_retention.csv`
- `metrics/bag_size_sweep_summary.csv`
- `metrics/quality_tier_sweep_summary.csv`
- `metrics/attention_pooling_instance_weights.csv`
- `metrics/run_manifest.json`

Required report:

- `reports/benign_variant_mil_report.md`

Required figures:

- `figures/mil_design_schematic.png`
- `figures/heldout_response_prediction_comparison.png`
- `figures/stability_vs_endpoint_pareto.png`
- `figures/benign_dispersion_pre_post_mil.png`
- `figures/pathogenic_ood_pre_post_mil.png`
- `figures/strong_pathogenic_retention_lines.png`
- `figures/bag_size_sweep.png`
- `figures/quality_tier_sweep.png`
- `figures/attention_weights_by_evidence_tier.png`
- `figures/retrieval_heatmaps_strong_pathogenic_pre_post.png`

## Proposed Implementation Order

1. Create the run root and write `configs/preflight_config.json`.
2. Stage the regenerated benign/control fixed-size table locally.
3. Validate required columns, stable row IDs, tier labels, sequence round trips,
   and exact wild-type amino-acid matches.
4. Build `benign_variant_bags_all.csv` with exact bag membership and tier labels.
5. Exclude Tier 5 from the primary `tier1_4` bag file.
6. Generate or verify ESM-C embeddings for all unique benign/control variant
   sequences, de-duplicating identical mutant sequences before embedding.
7. Validate that every bag instance has exactly one embedding row and that the
   wild-type embedding matches the canonical HopTF ESM-C row for that isoform
   within a documented numeric tolerance.
8. Stage or regenerate the pathogenic and matched-control variant panel.
9. Lock the strong pathogenic OOD panel from the current reference-only model.
10. Lock gene-level train/tune/hold splits.
11. Run `hoptf_reference_only` on the locked splits.
12. Run `hoptf_benign_instance_replicate`.
13. Run `hoptf_benign_mil_mean_pool`.
14. Run `hoptf_benign_mil_invariance`.
15. Select the provisional model family using tune response metrics plus benign
    stability only.
16. Run attention-pooling only if mean-pooling or invariance runs show promise
    under the non-pathogenic selection criteria.
17. Run bag-size sweep for the best simple model family.
18. Run quality-tier sweep.
19. Freeze the selected non-pathogenic configuration.
20. Evaluate pathogenic OOD retention once for the frozen configuration, then
    generate diagnostic OOD summaries for all prespecified tier sets.
21. Generate the final report and figures.

One-pass runbook gates:

- Gate A, staging complete: all required source files exist, checksums are
  recorded, and row-count expectations match the manifest.
- Gate B, joins complete: response metadata, wild-type ESM-C vocab, benign bag
  rows, and pathogenic panel rows all join by `isoform_embedding_id`.
- Gate C, embeddings complete: every sequence instance has one ESM-C embedding
  and no orphan embedding rows remain.
- Gate D, splits locked: no `gene_symbol` appears in more than one split.
- Gate E, reference baseline complete: reference-only metrics and benign
  dispersion have been written before MIL training starts.
- Gate F, non-pathogenic selection complete: selected model family and
  hyperparameters are frozen before pathogenic OOD evaluation.
- Gate G, OOD complete: strong pathogenic OOD retention is evaluated and written
  with missingness counts.

## Script Contract

Likely new scripts:

```text
scripts/build_benign_variant_bags.py
scripts/lock_strong_pathogenic_ood_panel.py
scripts/train_hoptf_benign_variant_mil.py
scripts/evaluate_benign_bag_stability.py
scripts/evaluate_pathogenic_ood_retention.py
scripts/summarize_benign_variant_mil.py
```

Preferred behavior:

- scripts should accept explicit paths and write all outputs under the run root;
- each run writes a JSON manifest with git commit, data paths, split seed, model
  seed, tier set, bag size, and loss weights;
- training configs should be saved as YAML or JSON before the run starts;
- never infer benign/control tier from row order.

## Statistical Tests

### Held-Out Response Prediction

- Paired bootstrap over held-out genes comparing MSE between reference-only and
  each MIL condition.
- Optional paired sign test over held-out genes for whether MIL improves more
  genes than it worsens.

### Benign Stability

- Paired bootstrap over isoforms comparing per-bag dispersion before and after
  MIL.
- Tier-stratified comparison of dispersion using gene-level bootstrap.

### Pathogenic OOD

- One-sided paired Wilcoxon comparing pathogenic OOD scores against matched
  benign/control OOD scores, within matched sets.
- One-sided paired Wilcoxon comparing pathogenic OOD scores against matched
  random-control OOD scores, if random controls are available.
- Gene-level bootstrap for median pathogenic-minus-control OOD difference.
- McNemar-style or paired proportion test for the fraction of strong pathogenic
  variants above the OOD threshold before versus after MIL, reported
  descriptively if sample size is small.

### Bag Size And Quality Trends

- Spearman correlation between bag size and stability improvement across
  run-level summaries.
- Spearman correlation between natural-evidence fraction and stability/OOD
  retention.
- Compare quality-prefix and random-subset policies using bootstrap over subset
  seeds and genes.

## Interpretation Rules

### If 10-Variant Bags Work Best

Interpretation:

- The fixed 10-control design provides useful tolerated sequence coverage.
- Report tier composition so the result is not misread as purely clinical
  benign evidence.

Needed caveat:

- A fixed-size bag can include lower-quality rows for isoforms with sparse
  natural evidence. Show that Tier 5 is not driving the result.

### If Fewer High-Quality Variants Work Better

Interpretation:

- Control quality matters more than bag size.
- Use high-quality fewer-row bags in the primary result and treat 10-row bags as
  a coverage-motivated engineering design.

### If Ortholog-Supported Tier 4 Helps

Interpretation:

- Evolutionarily supported substitutions can act as useful tolerated sequence
  augmentations.

Needed caveat:

- Tier 4 rows are synthetic tolerated controls, not clinical benign variants.

### If Tier 5 Fallback Helps

Interpretation:

- Treat as an engineering augmentation result only.
- Do not call it biological benign supervision.

### If Pathogenic OOD Collapses

Interpretation:

- MIL has taught the model to ignore too much sequence variation.
- Reduce stability loss weights, use higher-quality bags only, or add a
  non-pathogenic contrastive calibration set in a future experiment.

Do not present it as a successful MIL result.

## Figure Plan

### Figure 1: MIL Design Schematic

Show one TF isoform bag:

- wild-type sequence;
- Tier 1-4 tolerated/control variants;
- shared SCARF response target;
- HopTF query into AlphaGenome memory;
- bag pooling or stability regularization;
- held-out pathogenic variants evaluated after training.

Caption must state:

- tolerated/control variants share the isoform response label;
- pathogenic variants are held out;
- high-retrieval loci remain hypotheses, not binding sites.

### Figure 2: Endpoint-Stability Pareto

x-axis:

- held-out SCARF response MSE relative to reference-only.

y-axis:

- benign/tolerated predicted-response dispersion relative to reference-only.

Points:

- model condition and bag size.

Interpretation:

- The desired region is lower or equal MSE and lower benign dispersion.

### Figure 3: Bag Size Sweep

x-axis:

- number of tolerated/control variants per bag.

y-axis:

- benign dispersion reduction, held-out MSE change, and pathogenic OOD retention
  in separate panels.

Lines:

- quality-prefix subsets;
- random-subset mean with interval.

### Figure 4: Quality Tier Sweep

x-axis:

- tier set used for training.

y-axis:

- stability improvement and OOD retention.

This directly answers whether higher-quality benign annotations do better than
broader lower-quality bags.

### Figure 5: Pathogenic OOD Retention

x-axis:

- model condition: reference-only and selected MIL.

y-axis:

- OOD composite score.

Show:

- all pathogenic variants lightly;
- locked strong pathogenic variants with connected lines;
- OOD threshold line.

### Figure 6: Retrieval Examples Before And After MIL

For 2-3 locked strong pathogenic variants:

- WT retrieval;
- selected tolerated/control instance retrieval;
- pathogenic retrieval before MIL;
- pathogenic retrieval after MIL;
- pathogenic minus WT difference.

Use the same chromosome-binned heatmap convention as the current variant
retrieval figure.

## Paper Integration

If the experiment succeeds, add a Results subsection after the variant and
retrieval-movement results:

```text
Tolerance-aware MIL stabilizes benign sequence variation while preserving
pathogenic out-of-distribution responses
```

Suggested paper paragraph:

> We next trained HopTF with tolerated sequence-instance bags, where each TF
> isoform bag contained the reference sequence and up to 10 benign or tolerated
> variants sharing the same observed perturbation-response target. This
> multiple-instance objective reduced predicted-response and retrieval dispersion
> among tolerated variants relative to reference-only training. Importantly,
> pathogenic variants that produced strong baseline response or retrieval shifts
> remained out-of-distribution relative to the tolerated bag after MIL training,
> indicating that the learned invariance was selective rather than a collapse of
> sequence sensitivity.

If the result is stability-only:

> Benign-variant MIL stabilized tolerated sequence instances but did not improve
> held-out SCARF response prediction. We therefore treat the run as a regularized
> retrieval diagnostic rather than evidence that the current response model has
> stronger predictive performance.

If it fails:

> The tolerated-instance objective did not improve the current HopTF setup or
> over-regularized sequence sensitivity. This failure suggests that benign
> variant bags require stronger tier filtering, a weaker invariance objective, or
> additional contrastive supervision before they can be used as training
> evidence.

## Do Not Claim

- Do not claim HopTF predicts clinical pathogenicity.
- Do not claim every benign/control row is molecularly neutral.
- Do not pool Tier 1 clinical benign, Tier 2 population tolerated, Tier 3
  functional controls, Tier 4 ortholog-supported synthetic controls, and Tier 5
  fallback controls without tier-specific reporting.
- Do not claim a MIL win if instance replication performs equally well.
- Do not claim a MIL win if pathogenic OOD behavior collapses.
- Do not tune hyperparameters on pathogenic labels and then report pathogenic
  OOD retention as an independent result.
- Do not interpret high-retrieval AlphaGenome windows as validated binding sites.

## Minimal Viable Run

If time is very tight, run only:

1. Stage Tier 1-4 bags.
2. Embed benign/control variant sequences with ESM-C.
3. Lock the strong pathogenic OOD panel.
4. Compare:
   - `hoptf_reference_only`
   - `hoptf_benign_instance_replicate`
   - `hoptf_benign_mil_mean_pool`
   - `hoptf_benign_mil_invariance`
5. Use one locked gene split and three model seeds.
6. Evaluate:
   - held-out SCARF MSE;
   - benign predicted-response and retrieved-representation dispersion;
   - strong pathogenic OOD retention.

This minimal run is enough to decide whether the idea is alive. Bag-size and
quality sweeps can follow only if the minimal run passes the OOD retention guard.

## Open Implementation Risks

- The regenerated fixed-size benign/control table is referenced by the paper
  draft but is not present in this local workspace. It must be staged before
  implementation.
- ESM-C embedding generation for 32,640 control sequences may be the main runtime
  bottleneck unless embeddings already exist on the cluster or Hugging Face.
- The SCARF response artifacts must be available in a script-friendly format.
- Full attention JSD can be memory-heavy; if necessary, compute top-k
  approximations plus retrieved-vector distances.
- Some high-quality natural control bags will have fewer than 10 variants; do
  not fill them with weak controls unless the run is explicitly labeled
  fallback-inclusive.
- If response prediction remains dominated by protein length, ORF length, and
  recovered cell count, the best possible MIL result may be stability/OOD
  calibration rather than lower MSE.

## Done-Enough Criteria

The experiment is done enough to include in the report if:

- every model condition has a manifest and locked split;
- every bag row has evidence tier and source provenance;
- Tier 5 is excluded from the primary result or shown separately;
- held-out response metrics are compared to reference-only and artifact-only
  baselines;
- benign stability is reported before and after MIL;
- strong pathogenic OOD retention is reported before and after MIL;
- the final interpretation follows the decision table above.
