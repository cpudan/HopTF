#!/usr/bin/env bash
# Submit the HopTF ortholog-supported synthetic-control pipeline on the cluster.
#
# This wrapper assumes it is run from the HopTF repo on ne1-login. It writes all
# runtime artifacts under /gpfs/commons/groups/knowles_lab/dmeyer/hoptf and keeps
# final report-friendly CSV/JSON/FASTA outputs under data/processed in the repo.
set -euo pipefail

REPO=${REPO:-"$HOME/HopTF"}
RUN_ROOT=${RUN_ROOT:-/gpfs/commons/groups/knowles_lab/dmeyer/hoptf/ortholog_supported_controls_20260517}
OUTDIR=${OUTDIR:-"$REPO/data/processed/variant_controls/tf_isoform_ortholog_supported_controls_20260517"}
COMPARA=${COMPARA:-/gpfs/commons/groups/knowles_lab/data/cross_species/ensembl/ensembl_compara_115}
PEP_DIR=${PEP_DIR:-"$RUN_ROOT/ensembl_release_115_pep"}
PFAM=${PFAM:-/gpfs/commons/groups/knowles_lab/data/pfam_db/Pfam-A.hmm}
PARTITION=${PARTITION:-cpu}
TIME=${TIME:-12:00:00}
MEM=${MEM:-64G}
CPUS=${CPUS:-8}

mkdir -p "$RUN_ROOT/logs" "$OUTDIR"
SBATCH_FILE="$RUN_ROOT/run_ortholog_supported_controls.sbatch"

cat > "$SBATCH_FILE" <<SBATCH
#!/bin/bash
#SBATCH --job-name=hoptf_ortholog
#SBATCH --partition=$PARTITION
#SBATCH --time=$TIME
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --output=$RUN_ROOT/logs/run_ortholog_supported_controls.%j.out
#SBATCH --error=$RUN_ROOT/logs/run_ortholog_supported_controls.%j.err
set -euo pipefail

cd "$REPO"
PY="$REPO/.venv/bin/python"
if [ ! -x "\$PY" ]; then
  PY=python3
fi

if command -v uv >/dev/null 2>&1; then
  uv pip install --python "\$PY" --quiet --upgrade biopython pandas numpy
else
  "\$PY" -m pip install --quiet --upgrade biopython pandas numpy
fi

"\$PY" scripts/build_ortholog_supported_benign_controls.py \\
  --metadata data/processed/linear_probe/tfatlas_subsample/PERTURBATION_METADATA_hard_local_subsample.csv \\
  --compara-dir "$COMPARA" \\
  --pep-dir "$PEP_DIR" \\
  --outdir "$OUTDIR" \\
  --write-candidate-proteins-only

set +u
source /etc/profile.d/modules.sh 2>/dev/null || true
module load HMMER/3.4-gompi-2023a || module load hmmer/3.4-gompi-2023a || true
set -u
if ! command -v hmmscan >/dev/null 2>&1; then
  echo "hmmscan not available after module load" >&2
  exit 1
fi

hmmscan \\
  --cpu "$CPUS" \\
  --domtblout "$OUTDIR/ortholog_supported_candidate_proteins.pfam.domtblout" \\
  "$PFAM" \\
  "$OUTDIR/ortholog_supported_candidate_proteins.fasta" \\
  > "$OUTDIR/ortholog_supported_candidate_proteins.pfam.hmmscan.log"

"\$PY" scripts/parse_hmmscan_domtblout.py \\
  --domtblout "$OUTDIR/ortholog_supported_candidate_proteins.pfam.domtblout" \\
  --out "$OUTDIR/ortholog_supported_candidate_proteins.pfam_domains.tsv"

"\$PY" scripts/build_ortholog_supported_benign_controls.py \\
  --metadata data/processed/linear_probe/tfatlas_subsample/PERTURBATION_METADATA_hard_local_subsample.csv \\
  --compara-dir "$COMPARA" \\
  --pep-dir "$PEP_DIR" \\
  --outdir "$OUTDIR" \\
  --reuse-ortholog-table \\
  --pfam-domain-table "$OUTDIR/ortholog_supported_candidate_proteins.pfam_domains.tsv" \\
  --require-domain-architecture
SBATCH

jid=$(sbatch --parsable "$SBATCH_FILE")
echo "$jid" > "$RUN_ROOT/run_ortholog_supported_controls.jobid"
echo "submitted $jid"
