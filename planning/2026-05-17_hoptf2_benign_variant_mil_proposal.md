# HopTF2 Benign-Variant MIL Proposal

Date: 2026-05-17

## Objective

Train on Joung `H1 human embryonic stem cells`, learn a TF perturbation embedding from a wild-type TF plus benign missense variants, and measure local stability around the wild type under the existing HopTF retrieval path.

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

For this experiment, let $g$ index one wild-type TF background, let $B_g$ be the bag containing the wild-type sequence and its matched benign variants, and let $\mathrm{wt}(g)$ denote the wild-type member of that bag. For each sequence $i \in B_g$,
$$
q_i = W_q \, \mathrm{ESM\text{-}C}(s_i),
\qquad
\alpha_i = \mathrm{softmax}\!\left(\beta K_{\mathrm{AG}} q_i\right),
\qquad
\mu_i = V_{\mathrm{AG}}^\top \alpha_i.
$$

The teacher learns a small bag-specific basis for each bag,
$$
h_g^1, \dots, h_g^K \in \mathbb{R}^d,
\qquad
\pi_i \in \Delta^K,
\qquad
z_i = \sum_{k=1}^K \pi_i^k h_g^k,
$$
and the student predicts a latent $\hat z_i$ and endpoint $\hat y_i$ with target $y_i = \mathrm{latent\_pca}_i$. The first-pass objective is
$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{endpoint}}
+ \lambda_{\mathrm{teacher}} \mathcal{L}_{\mathrm{teacher}}
+ \lambda_{\mathrm{benign}} \mathcal{L}_{\mathrm{benign}},
$$
with
$$
\mathcal{L}_{\mathrm{endpoint}} = \ell(\hat y_i, y_i),
\qquad
\mathcal{L}_{\mathrm{teacher}} = \left\lVert \hat z_i - z_i \right\rVert_2^2,
\qquad
\mathcal{L}_{\mathrm{benign}} = \left\lVert \hat z_i - z_{\mathrm{wt}(g)} \right\rVert_2^2.
$$

This experiment enforces a shared sequence-side basis in AlphaGenome retrieval space for each wild-type TF and its benign missense variants, with benign variants anchored to the matched wild-type retrieval program.

## Experiment plan

Take the existing mutant panel outputs in `tmp/hoptf_variant_validation_expanded_20260516/`.

Read `variant_panel_mutant_sequences_metadata.csv`. Each table entry in this file represents one sequence example with fields including `isoform_embedding_id`, `mutation`, and `mutant_embedding_id`.

Keep only the wild-type table entries and the benign missense variant table entries.

Group table entries by `isoform_embedding_id`, so each bag is:

- one wild-type TF sequence
- its matched benign variants

Load wild-type `ESM-C` from the standard HopTF input.

Load mutant `ESM-C` from `variant_panel_mutant_esmc_embeddings.npy` with `variant_panel_mutant_esmc_vocab.json`.

Reuse `mutant_endpoint_predictions.csv` as the initial wild-type-versus-variant diagnostic surface.

Train a sequence-side MIL model where:

- the student is the HopTF retrieval path from `scripts/train_hopfield_projection.py`
- the teacher learns a small bag-specific basis for each wild-type-plus-benign bag
- benign variants are trained to match the matched wild-type latent
- the endpoint target is `latent_pca`

Compare three runs:

- `hoptf_baseline`
- `hoptf_benign_mut_mil_mean_pool`
- `hoptf_benign_mut_mil_attention`
