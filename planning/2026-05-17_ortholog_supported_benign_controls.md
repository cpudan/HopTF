# Ortholog-supported benign/control variant pipeline

Date: 2026-05-17

## Goal

Replace the earlier purely conservative Tier 4 fallback rows with a more defensible synthetic-control tier:

- the alternate amino acid must be observed at the aligned position in an Ensembl Compara ortholog;
- the substitution must pass conservative amino-acid chemistry filters;
- the position must avoid sequence edges, C2H2-like zinc-finger motifs, basic patches, leucine-zipper-like heptads, low-complexity windows, and Cys/His/Gly/Pro/Trp positions when possible;
- if Pfam annotations are available, the ortholog protein must have the same Pfam domain architecture as the mapped human Ensembl protein.

These rows are still not clinical benign variants. They should be reported as `Tier 4 ortholog-supported synthetic control` or as evolutionarily supported synthetic controls, and kept separate from ClinVar, gnomAD, and MaveDB evidence.

## Staged data

Existing cluster data:

- Ensembl Compara release 115:
  `/gpfs/commons/groups/knowles_lab/data/cross_species/ensembl/ensembl_compara_115`
- Pfam HMM database:
  `/gpfs/commons/groups/knowles_lab/data/pfam_db/Pfam-A.hmm`

New data downloaded by SLURM after email notice:

- Ensembl release 115 peptide FASTA for 13 vertebrates:
  `/gpfs/commons/groups/knowles_lab/dmeyer/hoptf/ortholog_supported_controls_20260517/ensembl_release_115_pep`
- Anticipated compressed size: about 142 MB.
- Actual on-disk size after download: 137 MB.
- Manifest:
  `/gpfs/commons/groups/knowles_lab/dmeyer/hoptf/ortholog_supported_controls_20260517/ensembl_release_115_pep/download_manifest.tsv`

Species panel:

- human, chimpanzee, gorilla, orangutan, macaque, mouse, rat, dog, cow, opossum, chicken, Xenopus tropicalis, zebrafish.

## Scripts

- `scripts/build_ortholog_supported_benign_controls.py`
  - maps HopTF isoforms to same-gene human Ensembl proteins;
  - extracts high-confidence orthologs from Ensembl Compara;
  - writes candidate protein FASTA for Pfam scanning;
  - generates HGVS-validated ortholog-supported synthetic substitutions.
- `scripts/parse_hmmscan_domtblout.py`
  - converts HMMER `hmmscan --domtblout` output into a compact protein-domain table.
- `scripts/run_ortholog_supported_controls_cluster.sh`
  - submits the full cluster pipeline.

## Full cluster command

Run from `ne1-login`:

```bash
cd ~/HopTF
RUN_ROOT=/gpfs/commons/groups/knowles_lab/dmeyer/hoptf/ortholog_supported_controls_20260517 \
OUTDIR=/gpfs/commons/groups/knowles_lab/dmeyer/hoptf/ortholog_supported_controls_20260517/full_outputs \
PARTITION=cpu \
TIME=18:00:00 \
MEM=96G \
CPUS=12 \
scripts/run_ortholog_supported_controls_cluster.sh
```

Current full-run job:

```text
16143826  hoptf_ortholog
```

## Expected outputs

Main output directory:

```text
/gpfs/commons/groups/knowles_lab/dmeyer/hoptf/ortholog_supported_controls_20260517/full_outputs
```

Expected files:

- `ortholog_supported_isoform_human_mapping.csv`
- `ortholog_supported_compara_ortholog_pairs.tsv`
- `ortholog_supported_candidate_proteins.fasta`
- `ortholog_supported_candidate_proteins.csv`
- `ortholog_supported_candidate_proteins.pfam.domtblout`
- `ortholog_supported_candidate_proteins.pfam_domains.tsv`
- `tf_isoform_ortholog_supported_controls_20260517.csv`
- `tf_isoform_ortholog_supported_controls_20260517.metadata_only.csv`
- `tf_isoform_ortholog_supported_controls_20260517.hgvs_parse_validation.csv`
- `tf_isoform_ortholog_supported_controls_20260517.fasta`
- `tf_isoform_ortholog_supported_controls_20260517.vocab.json`
- `tf_isoform_ortholog_supported_controls_20260517.sequences.json`
- `tf_isoform_ortholog_supported_controls_20260517.summary.json`

## Validation already run

A mocked mini-Compara smoke test passed on the cluster. It verified:

- HopTF isoform to human Ensembl protein mapping;
- Compara ortholog extraction;
- aligned ortholog-residue substitution nomination;
- HGVS generation and sequence round-trip validation;
- FASTA, vocab, sequence JSON, full CSV, metadata CSV, and summary JSON writing.

Smoke-test output:

```text
/gpfs/commons/groups/knowles_lab/dmeyer/hoptf/ortholog_supported_controls_20260517/smoke_test/out
```

## Important caveats

- The pipeline enforces ortholog residue support and optional Pfam architecture matching, but does not yet use curated residue-level TF contact/interface annotations. Known-contact exclusion is currently represented by sequence-window heuristics.
- Exact Pfam architecture matching is conservative and may reduce yield. If yield is too low, inspect the domain table before loosening this rule.
- Synthetic ortholog-supported controls should not be pooled with natural benign/control evidence without tier-specific reporting.
