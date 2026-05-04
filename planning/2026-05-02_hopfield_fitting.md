# Hopfield Fitting And OT-CFM Integration Plan

## Purpose

Build an overnight-testable version of the full sequence-to-cell-state system:

1. Load TF isoform ESM-C sequence embeddings.
2. Project each ESM-C vector through one fully connected layer into the AlphaGenome key space.
3. Fit Hopfield-style associations between each TF isoform sequence representation and AlphaGenome-derived keys.
4. Feed the same sequence representation, plus any Hopfield retrieval output, into a TorchCFM OT-CFM model that transports a control-cell latent representation to the TF-perturbed latent state.
5. Evaluate whether held-out TF isoforms with similar measured responses are predicted similarly, and whether sequence-only conditioning separates isoforms from the same TF that produce different responses.

The immediate target is not a polished model. The target is a set of small, instrumented component tests that can run overnight and tell us where the system is broken.

## Status Update: 2026-05-03

### Completed

- Implemented the overnight smoke-pipeline scripts:
  - `scripts/inspect_hopfield_inputs.py`
  - `scripts/prepare_alphagenome_keys.py`
  - `scripts/build_perturbation_latents.py`
  - `scripts/train_hopfield_projection.py`
  - `scripts/train_otcfm_sequence_conditioned.py`
  - `scripts/evaluate_isoform_holdouts.py`
  - `scripts/make_mutant_esmc_embeddings.py`
  - `scripts/run_hopfield_overnight.py`
  - shared helpers in `scripts/hopfield_fitting_common.py`
- Added focused smoke tests in `tests/test_hopfield_fitting_smoke.py`.
- Verified the repo test suite: `8 passed`.
- Ran the main overnight smoke runner successfully:
  - report: `tmp/hopfield_fitting/overnight_summary.md`
  - status: `10 / 10` runner steps passed.
- Confirmed the local input contract:
  - ESM-C embeddings are `3548 x 1152` `float32`.
  - metadata label counts are `816` responders, `816` nonresponders, `1632` ambiguous, and `2` `control_null`.
  - length/cell/response correlations were reproduced:
    - `protein_aa_length` vs `n_cells` Spearman rho `-0.7406`
    - `response_score` vs `n_cells` Spearman rho `-0.9024`
    - `protein_aa_length` vs `response_score` Spearman rho `0.6506`
- Downloaded the real Hugging Face AlphaGenome candidate archive from `H2dddhxh/AI4Genome` into `data/hf_min/external/AI4Genome/`.
- Prepared real AlphaGenome gene-pooled keys:
  - full archive: `tmp/hopfield_fitting_real_ag/alphagenome_keys.npy`
  - shape: `54901 x 3072`
  - report: `tmp/hopfield_fitting_real_ag/alphagenome_keys_report.json`
- Prepared chromosome 1-only AlphaGenome keys:
  - chr1 archive: `tmp/hopfield_fitting_real_ag_chr1/alphagenome_keys.npy`
  - shape: `5087 x 3072`
  - report: `tmp/hopfield_fitting_real_ag_chr1/alphagenome_keys_report.json`
- Confirmed there is not a separate chr1 file in the Hugging Face archive; chr1 is represented as filterable rows in the pooled archive.
- Trained Hopfield projection smoke tests:
  - synthetic retrieval passed with top-1 accuracy `1.0`.
  - real overfit with synthetic keys passed with top-1 accuracy `1.0`.
  - real overfit against the real 3072-d AlphaGenome key space passed after tuning with top-1 accuracy `0.9531`.
- Built perturbation-level PCA endpoint latents from the published TF Atlas `.h5ad`:
  - output: `tmp/hopfield_fitting/perturbation_latents.npz`
  - metadata: `tmp/hopfield_fitting/perturbation_latents_metadata.csv`
- Installed and exercised TorchCFM:
  - `torchcfm_used=True` in the CFM metrics.
  - synthetic OT-CFM smoke test passed.
  - real perturbation endpoint overfit passed.
  - HNF4A leave-one-isoform smoke test passed for `2 / 2` heldouts.
- Ran baseline isoform-sibling evaluation for `HNF4A`, `TP53`, and `NFATC1`:
  - output: `tmp/hopfield_fitting/isoform_holdout_baselines.csv`
  - summary: `tmp/hopfield_fitting/isoform_holdout_baselines_summary.json`
- Generated coordinate-checked mutant FASTA/metadata:
  - FASTA: `tmp/hopfield_fitting/mutant_sequences.fasta`
  - metadata: `tmp/hopfield_fitting/mutant_sequences_metadata.csv`
  - verified mutants: HNF4A-7 `R85W`, `S87N`, `R89W`, `T139I`; GATA2-2 `R396Q`, `R362Q`.
- Correctly skipped TP53 canonical hotspot mutations for the current TP53 responder isoform because local isoform coordinates did not match canonical TP53 residue positions.
- Sent a writeup email with the run summary.
- Updated `scripts/run_hopfield_overnight.py` so real AlphaGenome keys are a first-class runner mode:
  - `--key-source auto|synthetic|gene_pooled_npz`
  - `--alphagenome-npz data/hf_min/external/AI4Genome/gene_pooled_embeddings.npz`
  - repeatable `--holdout-gene`
  - `--max-holdouts-per-gene`
- Ran the updated real-key runner successfully:
  - command: `uv run python scripts/run_hopfield_overnight.py --outdir tmp/hopfield_fitting_flight --key-source auto --holdout-gene HNF4A --holdout-gene TP53 --max-holdouts-per-gene 1`
  - report: `tmp/hopfield_fitting_flight/overnight_summary.md`
  - status: `12 / 12` runner steps passed.
  - key source: `gene_pooled_npz`
  - trained holdout genes: `HNF4A`, `TP53`
- Added unified endpoint baseline reporting:
  - script: `scripts/evaluate_sequence_endpoint_baselines.py`
  - report: `tmp/hopfield_fitting_flight/sequence_endpoint_baselines.md`
  - summary: `tmp/hopfield_fitting_flight/sequence_endpoint_baselines_summary.json`
- Added controlled sequence baseline reporting:
  - script: `scripts/evaluate_controlled_sequence_baselines.py`
  - report: `tmp/hopfield_fitting_flight/controlled_evaluation_report.md`
  - summary: `tmp/hopfield_fitting_flight/controlled_sequence_baselines_summary.json`
  - matched contexts: overall grouped, length-matched, cell-count-matched, and combined length-cell-matched.
- Baseline report currently shows length/cell/ORF covariates beating ESM-C and Hopfield-query ridge models under grouped gene splits:
  - best baseline: `length_cell_orf`, MSE `6.5946`
  - `hopfield_query`, MSE `6.8686`
  - `esm_c`, MSE `7.2760`
  - shuffled controls are worse: `8.6509` and `8.8106`
- Controlled report currently shows that adding Hopfield-query features to artifact covariates improves endpoint prediction, while adding raw ESM-C does not:
  - overall artifact-only MSE `6.5946`
  - overall artifact+Hopfield-query MSE `6.2728`, a `4.88%` improvement
  - length-matched artifact+Hopfield-query improvement `4.32%`
  - cell-count-matched artifact+Hopfield-query improvement `2.73%`
  - combined length-cell-matched artifact+Hopfield-query improvement `1.69%`
  - artifact+ESM-C is worse than artifact-only in all matched contexts tested so far.
- Re-ran the updated real-key runner with controlled reporting included:
  - report: `tmp/hopfield_fitting_flight/overnight_summary.md`
  - status: `13 / 13` runner steps passed.
- Probed NFATC1 as a trained leave-one-isoform CFM check:
  - output: `tmp/hopfield_fitting_nfatc1_probe/otcfm_leave_one_NFATC1_summary.json`
  - status: failed strict baseline-beating criterion, endpoint MSE fraction of baseline `1.1854`
  - interpretation: useful failing case for debugging, not ready as a required green-path runner step.

### Partially Completed

- AlphaGenome key integration works, including real 3072-d keys and chr1 filtering. The runner can now use real keys directly. Synthetic mode remains available for minimal smoke tests when the downloaded archive is absent.
- TorchCFM integration is real and passing smoke tests, but the current model uses a simple MLP conditioning path over PCA centroid endpoints. It is not yet a production-quality cell-level transport model.
- Held-out isoform evaluation exists. Trained one-heldout CFM smoke checks pass for `HNF4A` and `TP53`; `NFATC1` currently fails and should be treated as a debugging target. The broader mixed-response TF panel still needs systematic runs.
- Mutation sequence generation and coordinate validation exist, but mutant ESM-C embeddings have not been regenerated because no local ESM-C model snapshot/weights were present.
- The response artifact logic was checked and reproduced. Grouped ridge and controlled matched-strata reports now include length/cell/ORF, sequence, Hopfield-query, and shuffled controls. Downsampling and uncertainty weights are not yet integrated into every training/evaluation report.

### Remaining

- Extend the unified and controlled model-comparison reports into a single combined table that includes trained Hopfield+OT-CFM heldout metrics on the same split definitions.
- Expand leave-one-isoform training/evaluation beyond the current passing `HNF4A` and `TP53` smoke checks to `ZNF195`, `IKZF3`, `TP73`, `MIER1`, `SOX5`, `ZNF534`, and other mixed-response TF groups.
- Debug `NFATC1`, which currently fails the strict trained leave-one-isoform baseline-beating criterion.
- Add length-matched and cell-count-matched holdout splits.
- Add explicit covariates or controls for `log1p(n_cells)`, `protein_aa_length`, `orf_nt_length`, batch/library, and measurement uncertainty.
- Regenerate ESM-C embeddings for verified mutant sequences once the local ESM-C model snapshot/weights are available.
- Run frozen-model mutant endpoint predictions for HNF4A and GATA2 mutants, then compare deleterious vs benign/weak-effect substitutions.
- Resolve TP53 mutation coordinate mapping by choosing the correct canonical isoform/offset or a local isoform-specific mutation panel.
- Curate exact FOS/JUN/AP-1 DNA-binding substitutions with direct functional evidence and add them to the mutation panel.
- Add endpoint plots in PCA space and predicted-vs-observed response plots.
- Add differential-expression or signature-agreement metrics where pseudobulk counts are available.
- Move beyond perturbation-level PCA centroids:
  - add SCARF latent endpoints,
  - then add cell-level mini-batch training/evaluation.
- Verify AlphaGenome genome-build provenance explicitly. The current archive metadata exposes coordinates and chromosomes but does not by itself prove GRCh38/hg38.
- Decide whether generated large artifacts under `tmp/hopfield_fitting*` and downloaded `data/hf_min/external/AI4Genome/` should stay local only, be ignored, or be promoted to a tracked/reproducible data download step.

## Current Repo Facts

- Local ESM-C matrix:
  `data/hf_min/processed/protein_embeddings/tf_atlas_morf_isoforms_esmc_600m/tf_atlas_morf_isoforms_esmc_600m_mean_non_special.npy`
  - shape observed locally: `(3548, 1152)`
  - dtype observed locally: `float32`
- Local ESM-C vocab:
  `data/hf_min/processed/protein_embeddings/tf_atlas_morf_isoforms_esmc_600m/metadata/tf_atlas_morf_isoform_vocab.json`
  - length observed locally: `3548`
  - first key pattern: `TFORF0001-HIF3A-HIF3A-1`
- Local response metadata:
  `data/processed/linear_probe/tfatlas_subsample/PERTURBATION_METADATA_hard_local_subsample.csv`
  - rows observed locally: `3266`
  - non-control perturbations: `3264`
  - responder/nonresponder/ambiguous split: `816 / 816 / 1632`
- Existing length-response QC outputs:
  `tf_length_response_qc_outputs/`
  - `protein_aa_length` vs `n_cells` Spearman rho observed locally: about `-0.741`
  - `response_score` vs `n_cells` Spearman rho observed locally: about `-0.902`
  - `protein_aa_length` vs `response_score` Spearman rho observed locally: about `0.651`
  - prior conclusion: partially coverage-driven, residual length effect remains.
- `origin/audrey` currently does not contain TorchCFM or diffusion code. It contains the AlphaGenome notebook and a lighter AlphaGenome dependency set. We should not assume hidden CFM code exists there.
- Existing `scripts/generate_k_matrix.py` generates an AlphaGenome `CHIP_TF` prediction matrix over genomic loci. That is useful, but it is not the same thing as the recently observed Hugging Face `gene_pooled_embeddings.npz` AlphaGenome embedding archive. For the query/key compatibility test below, prefer 3072-d AlphaGenome embeddings if available; otherwise use the existing CHIP_TF K matrix path as a separate phenotype-key ablation.

## Hugging Face AlphaGenome Artifact Check

The best recent Hugging Face candidate found was `H2dddhxh/AI4Genome`, last modified March 2026. It contains:

- `gene_pooled_embeddings.npz`, about 569 MB, pushed March 24, 2026.
- `ckpt_alphagenome_pretrain_16gpu_2node_mean_epoch47.npz`, about 99 MB, pushed March 21, 2026.
- `README_gene_pooled_embeddings.txt`, describing the NPZ as AlphaGenome gene-level pooled DNA embeddings with shape `(G, 3072)`.

The README describes 128 bp windows pooled per gene, with fields like `pooled`, `gene_ids`, `gene_symbols`, `chroms`, `starts`, `ends`, `strands`, and `seq_lengths`. It does not explicitly state GRCh38/hg38 in the visible metadata, so the loader should treat genome build as an assertion to verify from coordinates or upstream provenance rather than as known fact.

No separate chromosome 1 or `chr1` AlphaGenome embedding file was found in that repository or in the Hugging Face searches I ran. Plan around loading the single gene-pooled archive and filtering rows where `chroms == "chr1"` if chromosome 1-specific tests are needed.

## Logic Check: Protein Length, Cell Counts, And Response

The stated artifact model is logically sound, with one caveat about causal interpretation.

Working causal story:

- Longer ORFs are harder to package/transduce in large lentiviral libraries, so longer proteins tend to have fewer recovered cells.
- TFs that create strong phenotypic responses can reduce proliferation, viability, or recoverability, so strong responders also tend to have fewer recovered cells.
- Since both protein length and response strength push against cell recovery, protein length becomes correlated with response score in the observed dataset.

The caveat: `n_cells` is not a pure confounder. It can be downstream of both ORF length and biological response, and it is also tied to measurement noise. Regressing out `n_cells` alone can induce collider/overadjustment problems. We should use multiple controls:

- length-matched splits,
- downsampling to equal cell counts,
- response-score uncertainty estimates or weights based on `n_cells`,
- explicit covariates for `log1p(n_cells)`, `protein_aa_length`, `orf_nt_length`, and batch/library,
- negative-control endpoints where length should not help,
- held-out isoform-group evaluation where gene-level identity shortcuts are blocked.

For overnight debugging, every metric should be reported both raw and length/cell-count controlled.

## System Architecture

### Sequence Encoder And Projection

Inputs:

- `e_i`: ESM-C embedding for isoform `i`, shape `[1152]`.
- `K`: AlphaGenome key matrix, shape `[M, d_key]`.

Projection:

```text
q_i = LayerNorm(W_q e_i + b_q)
```

where `W_q` is a single fully connected layer from `1152 -> d_key`.

If using the Hugging Face AlphaGenome gene-pooled embedding archive, `d_key = 3072`.
If using the current local CHIP_TF prediction K matrix, `d_key = n_chip_tf_tracks`, and the plan should call that a phenotype-key ablation rather than a direct AlphaGenome embedding-key test.

### Hopfield Association Layer

Use modern Hopfield retrieval as attention over fixed keys:

```text
a_i = softmax(beta * q_i K^T)
r_i = a_i V
```

Where:

- `K`: AlphaGenome-derived key matrix.
- `V`: value matrix. Overnight candidates:
  - same as `K` for an auto-associative smoke test,
  - gene-level or locus-level target summaries,
  - learned values that map to response deltas in cell-latent space.
- `beta`: inverse temperature. Start fixed; later learn or schedule it.
- `r_i`: retrieved sequence-conditioned representation that goes to the downstream flow model.

Minimum component tests:

- shape test for `ESM-C -> Linear -> q`;
- key normalization test: L2 normalize `q` and `K`, inspect dot-product distribution;
- retrieval sanity test: for synthetic pairs, verify that the query retrieves the planted key;
- overfit test: on 8-16 isoforms, train until retrieval loss and endpoint loss go down;
- permutation test: shuffle isoform-to-response labels and confirm metrics collapse.

### OT-CFM Diffusion/Flow Model

Use TorchCFM as a conditional flow-matching adapter. TorchCFM documents OT-CFM as using a static optimal transport plan and optimal probability paths/vector fields to approximate dynamic OT. The original OT-CFM paper states that CFM does not require a Gaussian source, and that OT-CFM creates simpler, more stable flows with faster inference.

Training pair:

- `x0`: control-cell latent.
- `x1`: TF-perturbed cell latent or perturbation pseudobulk latent.
- `c_i`: sequence condition for TF isoform `i`, e.g. `[q_i, r_i, protein_length_controls_optional]`.

Overnight version:

- Use existing PCA or SCARF/HVG latent as `x`.
- Start with pseudobulk centroids per perturbation to remove single-cell noise.
- Then expand to cell-level mini-batches after the pseudobulk endpoint test passes.
- Use OT-CFM mini-batch matching between source control latents and target perturbed latents.

Vector field signature:

```text
v_theta(t, x_t, c_i) -> dx/dt
```

Conditioning strategy:

- concatenate `[x_t, time_embedding(t), c_i]` into an MLP for the first overnight run;
- later replace this with FiLM or cross-attention if the simple conditioning overfits but does not generalize.

Sampling/evaluation:

- start from held-out control cells or the control centroid;
- integrate to `t=1`;
- compare predicted endpoint to observed held-out perturbation latent.

## Held-Out Isoform Tests

Primary split should be leave-one-isoform-out within a TF gene, not random cell splits. Random splits leak isoform identity and response structure.

Candidate groups from local metadata:

- `TP53`: 12 isoforms, mixed response labels; 1 responder, 8 nonresponders, 3 ambiguous. Strong for same-gene/different-response testing, but the responder currently has only 11 cells, so it is a smoke test unless more cells are available elsewhere.
- `HNF4A`: 7 isoforms; `HNF4A-7` is responder with 19 cells, `HNF4A-5` and `HNF4A-6` are nonresponders, others ambiguous. Good biological mutation-control candidate, but cell counts are still modest for the responder.
- `NFATC1`: 11 isoforms, 5 responders, 2 nonresponders, 4 ambiguous, min cells 5. Good for mixed-response splits if low-cell rows are filtered or downsampled consistently.
- `ZNF195`: 7 isoforms, 5 responders, 1 nonresponder, 1 ambiguous. Good for mixed-response testing, but zinc-finger isoforms need careful sequence/domain interpretation.
- `CAMTA2`: 4 isoforms, all responders. Useful for similar-response clustering, but not for separating same-TF different-response isoforms.
- `TP73`, `IKZF3`, `MIER1`, `SOX5`, `ZNF534`, `ZNF568`: additional mixed groups for robustness.

Evaluation metrics:

- endpoint L2/cosine distance in latent space;
- rank of the true held-out perturbed centroid among candidate perturbation centroids;
- same-response sibling distance vs different-response sibling distance;
- responder/nonresponder classification from predicted endpoint response score;
- agreement of top differential-expression/signature scores where pseudobulk counts are available;
- calibration by `n_cells`: bin predictions by held-out cell count and report uncertainty.

Pass criteria for overnight:

- synthetic Hopfield retrieval overfits planted associations;
- real-data 8-16 isoform subset overfits;
- held-out isoform predictions are closer to same-response siblings than to different-response siblings in at least one mixed TF group;
- shuffled sequence embeddings and shuffled labels fail;
- length-only baseline is explicitly reported and does not explain all held-out performance.

## Mutation Control Tests

Goal: test whether sequence-conditioned predictions respond to biologically meaningful amino-acid substitutions, not just TF identity or protein length.

### Candidate 1: HNF4A

Local data has an HNF4A responder isoform (`HNF4A-7`, 19 cells) and HNF4A nonresponders (`HNF4A-5`, `HNF4A-6`).

Literature-backed mutations:

- deleterious/control-loss candidates: `R85W`, `S87N`, `R89W` in HNF4A isoform 2 numbering. A 2024 functional study reports reduced DNA binding for `R85W`, `S87N`, and `R89W`; the table reports HNF1A-promoter DNA binding of 35%, 13%, and 9% of wild type, and G6PC-promoter binding of about 10% for `R85W`.
- benign/weak-effect candidate: `T139I` in the same study is treated as a benign-with-respect-to-MODY control and has BS3-supporting functional evidence, with transactivation above 75% in the reported assays.

Overnight test:

- mutate the exact HNF4A isoform sequence only if residue numbering maps cleanly to the local isoform sequence;
- regenerate ESM-C embeddings for WT and mutant sequences;
- pass WT, deleterious mutants, and benign control through the frozen/fit projection plus Hopfield/CFM model;
- expected result: deleterious DBD mutants move predicted endpoint toward weak/no response more than `T139I` or synonymous/no-op controls.

### Candidate 2: TP53

Local data has many TP53 isoforms and strong same-gene response diversity, but the responder has only 11 cells. Use as a mechanistic smoke test, not the first statistically powered result.

Literature-backed mutations:

- deleterious/control-loss candidates: `R248Q`, `R248W`, `R273H`. These are DNA-contact or DNA-binding-domain hotspot mutations. A Science study reports that `R175H`, `R248Q`, and `R273H` lost nearly all DNA-binding activity in ChIP-seq; structural reviews describe `R248` and `R273` as direct DNA-contact residues.
- benign/WT-like candidate: `R342Q`. ClinVar summarizes experimental functional studies showing DNA binding, tetramerization, transcriptional activation, and growth suppression similar to wild-type TP53.

Overnight test:

- use TP53 for the mutation machinery and endpoint-direction sanity checks;
- require a separate powered run or more cells before claiming TP53 biology from this dataset.

### Candidate 3: GATA2

Local data has `GATA2-2` as a responder, but only 10 cells.

Literature-backed mutations:

- deleterious/control-loss candidate: `R396Q`. Recent GATA2 work reports notably reduced DNA-binding and transactivation capabilities; structural discussion says R396 is essential for recognizing the WGATAR motif.
- cautious benign/weak-effect candidate: `R362Q` is described in one JCI model as likely having little effect on DNA binding because R362 is solvent exposed and makes only minor phosphate interactions. However, ClinVar/later reports may classify or functionally interpret this variant differently, so use it only as a weak-effect control with a warning.

Overnight test:

- use GATA2 only if residue mapping is clean and cell-count filtering does not discard the responder.

### Candidate 4: FOS/JUN/AP-1

Local FOS has a responder with 68 cells, making it attractive. The AP-1 basic region is well established as the DNA-binding region, and Fos/Jun basic regions contact both strands of the AP-1 site. However, we should not use this as the first mutation-control pair until we pin down exact human FOS substitutions with direct functional evidence.

Overnight action:

- add a curation task to identify exact FOS/JUN basic-region substitutions that abolish AP-1 DNA binding and a conservative/benign substitution control.

## Overnight Implementation Checklist

### 1. Data Contract Script

Create `scripts/inspect_hopfield_inputs.py`.

Inputs:

- ESM-C `.npy`
- ESM-C vocab `.json`
- perturbation metadata `.csv`
- AlphaGenome key `.npz` or `.npy`

Outputs:

- `tmp/hopfield_fitting/input_contract.json`
- shape/dtype checks;
- missing-vocab rows;
- number of responder/nonresponder/ambiguous isoforms;
- candidate held-out isoform groups after `min_cells` filtering;
- Spearman/correlation summary for length, cell counts, and response.

Debug target:

- this should run in under 1 minute and fail loudly on schema mismatches.

### 2. AlphaGenome Key Loader

Create `scripts/prepare_alphagenome_keys.py`.

Modes:

- `--source gene_pooled_npz`: load `pooled`, `gene_ids`, `gene_symbols`, `chroms`, `starts`, `ends`, `strands`, `seq_lengths`; output normalized key tensor and metadata.
- `--source chip_tf_k_matrix`: load local/generated K matrix from `scripts/generate_k_matrix.py`; output normalized key tensor and metadata, clearly marked as CHIP_TF phenotype keys.
- `--source synthetic`: make a small synthetic key matrix for smoke tests.

Outputs:

- `tmp/hopfield_fitting/alphagenome_keys.npy`
- `tmp/hopfield_fitting/alphagenome_keys_metadata.csv`
- `tmp/hopfield_fitting/alphagenome_keys_report.json`

Debug target:

- support synthetic keys immediately, so Hopfield and CFM tests can proceed even if the full AlphaGenome archive is not local yet.

### 3. Hopfield Module Smoke Test

Create `scripts/train_hopfield_projection.py`.

Minimum module:

```python
class SequenceToHopfield(nn.Module):
    def __init__(self, esm_dim, key_dim):
        self.query = nn.Sequential(nn.Linear(esm_dim, key_dim), nn.LayerNorm(key_dim))

    def forward(self, esm, keys, values=None):
        q = F.normalize(self.query(esm), dim=-1)
        k = F.normalize(keys, dim=-1)
        logits = beta * q @ k.T
        attn = logits.softmax(dim=-1)
        return attn @ (keys if values is None else values), attn, q
```

Losses:

- synthetic retrieval loss for planted query-key pairs;
- real-data endpoint loss if values are response-delta latents;
- entropy penalty optional, to avoid uniform retrieval.

Outputs:

- `tmp/hopfield_fitting/hopfield_smoke_metrics.json`
- attention entropy histograms;
- top-k key reports per isoform;
- saved checkpoint.

Pass/fail:

- synthetic overfit must pass before running real data;
- real-data tiny overfit should reduce loss sharply on 8-16 isoforms.

### 4. Pseudobulk Latent Endpoint Builder

Create `scripts/build_perturbation_latents.py`.

Inputs:

- `.h5ad` with PCA/SCARF/HVG latent or raw counts;
- perturbation column;
- metadata with `n_cells`, `response_score`, `label_status`.

Outputs:

- control latent distribution;
- perturbation centroid latent per TF isoform;
- per-isoform uncertainty estimates based on cell count;
- optional pseudobulk count matrix for signature/DE checks.

Implementation note:

- start with existing PCA latent under the response-score pipeline, because it is already used for local QC.
- add SCARF latent later once the endpoint tests pass.

### 5. TorchCFM Adapter Smoke Test

Create `scripts/train_otcfm_sequence_conditioned.py`.

Dependencies:

- add a separate optional env or dependency group for `torch`, `torchcfm`, and `pot`/OT backend if needed.
- do not mix this into the AlphaGenome environment until dependency conflicts are known.

Training stages:

1. synthetic 2D or 16D Gaussian transport conditioned on fake sequence vectors;
2. real pseudobulk latent overfit on 8-16 isoforms;
3. leave-one-isoform-out within candidate TF groups;
4. mutation embedding sanity check with frozen projection/flow.

Outputs:

- `tmp/hopfield_fitting/otcfm_synthetic_metrics.json`
- `tmp/hopfield_fitting/otcfm_overfit_metrics.json`
- `tmp/hopfield_fitting/otcfm_leave_isoform_out_metrics.csv`
- endpoint plots in PCA space;
- predicted vs observed response score plots, raw and length/cell-count controlled.

Pass/fail:

- synthetic transport must pass before real data;
- real overfit must pass before held-out evaluation;
- held-out performance must beat length-only and shuffled-sequence baselines.

### 6. Held-Out Isoform Evaluation

Create `scripts/evaluate_isoform_holdouts.py`.

Required split modes:

- `leave_one_isoform_out_by_gene`
- `leave_one_responder_out_by_gene`
- `same_response_vs_different_response_sibling`
- `length_matched_holdout`

Candidate first-pass groups:

- `TP53`, `HNF4A`, `NFATC1`, `ZNF195`, `IKZF3`, `TP73`, `MIER1`, `SOX5`, `ZNF534`.

Report:

- group, held-out isoform, cells, length, response score, label;
- predicted endpoint distance to true held-out endpoint;
- nearest observed perturbations;
- same-response sibling rank;
- different-response sibling rank;
- length-only baseline;
- shuffled-label baseline.

### 7. Mutation Embedding Builder

Create `scripts/make_mutant_esmc_embeddings.py`.

Inputs:

- metadata row or isoform embedding ID;
- list of mutations using protein coordinates;
- ESM-C snapshot path;
- existing sequence map.

Checks:

- assert residue at requested coordinate matches the literature WT residue;
- write both mutant FASTA and mutant embedding metadata;
- support no-op/synonymous and random conservative substitutions as controls.

Initial mutation panel:

- HNF4A: `R85W`, `S87N`, `R89W`, `T139I` if coordinate mapping matches local isoform.
- TP53: `R248Q`, `R248W`, `R273H`, `R342Q` if coordinate mapping matches local isoform.
- GATA2: `R396Q`, cautious `R362Q` if coordinate mapping matches local isoform.

Pass/fail:

- if residue mapping fails, skip that mutation and write a structured failure row instead of silently mutating the wrong residue.

## Overnight Run Order

1. Run data contract script.
2. Prepare synthetic AlphaGenome keys.
3. Train Hopfield projection on synthetic retrieval.
4. Build pseudobulk/control latent endpoints.
5. Train Hopfield projection on tiny real-data overfit.
6. Train OT-CFM synthetic transport.
7. Train OT-CFM tiny real-data overfit.
8. Run one leave-one-isoform-out evaluation on `HNF4A` and one on `TP53` or `NFATC1`.
9. Generate HNF4A mutant embeddings if coordinate mapping is clean.
10. Run mutation endpoint prediction with frozen trained model.
11. Write a single summary report under `tmp/hopfield_fitting/overnight_summary.md`.

## Debugging Priorities

- Fail fast on shape/key mismatches.
- Keep synthetic tests in every script, so errors are model-code bugs rather than biology/data bugs.
- Log every split and held-out isoform explicitly.
- Always report length-only, cell-count-only, ESM-C-only, Hopfield-only, and Hopfield+OT-CFM variants.
- Treat low-cell responder mutation tests as qualitative until they clear a minimum-cell threshold.

## References For Mutation And OT-CFM Choices

- HNF4A variants: Kaci et al., "Functional characterization of HNF4A gene variants identify promoter and cell line specific transactivation effects", Human Molecular Genetics, 2024. Reports reduced DNA binding for R85W/S87N/R89W and BS3-supporting evidence for T139I. https://academic.oup.com/hmg/article/33/10/894/7618438
- TP53 loss-of-DNA-binding variants: Boettcher et al., Science 2019, reports R175H/R248Q/R273H lost nearly all DNA-binding activity in ChIP-seq. https://labs.dana-farber.org/ebertlab/sites/g/files/prcqxy371/files/2026-01/Boettcher%2C%20et%20al.%20Science%202019.pdf
- TP53 benign/WT-like variant: ClinVar VCV000233136, R342Q, summarizes experimental studies showing DNA binding, tetramerization, transcriptional activation, and growth suppression similar to wild-type TP53. https://www.ncbi.nlm.nih.gov/clinvar/variation/233136/
- GATA2 R396Q: Blood Cancer Journal 2025 and JCI 2015 describe reduced DNA binding/transactivation and structural disruption of DNA interactions. https://www.nature.com/articles/s41408-025-01213-z and https://www.jci.org/articles/view/78888
- TorchCFM/OT-CFM: TorchCFM repo and Tong et al. OT-CFM paper. https://github.com/atong01/conditional-flow-matching and https://arxiv.org/abs/2302.00482
