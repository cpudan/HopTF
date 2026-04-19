# HopTF

This repository now uses three separate `uv` environments:

- The root project is the main environment for dataset conversion, local utilities, and lightweight tests.
- [`envs/scarf/pyproject.toml`](/home/dmeyer/courses/clmm/HopTF/envs/scarf/pyproject.toml) is a dedicated environment for the cbmi-group SCARF foundation model stack.
- [`envs/alphagenome/pyproject.toml`](/home/dmeyer/courses/clmm/HopTF/envs/alphagenome/pyproject.toml) is a separate `uv` project for the much heavier AlphaGenome stack.

Separating them avoids dependency conflicts between the SCARF foundation model, AlphaGenome/TensorFlow, and the lighter default tooling. It also prevents accidental installation of the unrelated PyPI package named `scarf`.

## Data

Project data is hosted on Hugging Face at:

- <https://huggingface.co/datasets/pvd232/HopTF>

## SCP3357 Conversion

The repository includes [`scripts/build_scp3357_h5ad.py`](./scripts/build_scp3357_h5ad.py) to turn the Single Cell Portal `SCP3357` download into a raw-counts `.h5ad` with aligned cell metadata, UMAP coordinates, and TF construct sequence metadata.

Download the study bundle from:

- <https://singlecell.broadinstitute.org/single_cell/study/SCP3357>

Place the extracted files under `data/raw/scp3357/`, or point the script at another location with `--input-dir`. The converter expects these paths from the SCP bundle:

- `metadata/metadata.tsv`
- `cluster/cluster.txt`
- `expression/*/counts_matrix.mtx`
- `expression/*/counts_barcodes.tsv`
- `expression/*/counts_features.tsv`

Run the default conversion with:

```bash
uv run python scripts/build_scp3357_h5ad.py
```

Or use an explicit input/output location:

```bash
uv run python scripts/build_scp3357_h5ad.py \
  --input-dir /path/to/SCP3357 \
  --output data/processed/scp3357/SCP3357_raw_counts.h5ad
```

The script also downloads the Addgene MORF workbook on demand if it is not already present at `data/reference/200102_tf_orf_library.xlsx`:

- <https://media.addgene.org/cms/filer_public/5e/22/5e22c6a5-d186-4c54-95c0-8314db54bfbe/200102_tf_orf_library.xlsx>

Outputs:

- `data/processed/scp3357/SCP3357_raw_counts.h5ad`
- `data/processed/scp3357/SCP3357_tf_construct_candidates_aa.fasta`

Key AnnData fields:

- `X` contains the raw UMI-corrected counts from `counts_matrix.mtx`.
- `layers["counts"]` stores a copy of the same raw count matrix for downstream tools that expect counts in a named layer.
- `obs` contains the portal metadata plus `perturbation_id`, which mirrors `tf_isoform_unique` and is the stable lookup key for the sequence tables.
- `obsm["X_umap"]` contains the portal UMAP coordinates from `cluster.txt`.
- `uns["tf_perturbation_map"]` maps each `perturbation_id` to its gene symbol, candidate construct IDs, and mapping status.
- `uns["tf_construct_candidates"]` stores the candidate MORF construct metadata, nucleotide ORFs, and translated amino acid sequences.
- `uns["tf_sequence_resources"]` records the lookup keys, source workbook, FASTA filename, and the ambiguity caveat.

Important caveat: some SCP3357 perturbation labels correspond to genes with multiple candidate MORF constructs in the public workbook. For those labels, the `.h5ad` stores the candidate set rather than claiming an exact construct-level assignment, because the SCP bundle does not expose MORF construct IDs or dial-out barcodes.

In the current SCP3357 mapping, this affects 11 of the 24 perturbation labels across 6 genes (`HLF`, `IRF2`, `MAFF`, `RELB`, `RORC`, and `SPI1`). Those ambiguous labels cover 4,644 of the 10,522 cells in the dataset.

Planned downstream handling:

- We will keep the full candidate set in the `.h5ad` as the primary provenance record.
- As part of the analysis, we will also try to infer which candidate construct sequence is most likely for each ambiguous perturbation label.
- We will then evaluate sequence-dependent results using that best-supported candidate assignment, while keeping the ambiguity explicit so we can revisit or sensitivity-check those calls if needed.

## SCARF Environment

The SCARF code used in this repo comes from the official foundation-model project:

- GitHub: <https://github.com/cbmi-group/scarf>
- Weights: <https://zenodo.org/records/17205044>
- Vendored encoder snapshot: commit `1b51ea8f9e861308b103d5dc70df20cf785626ce`

The minimal encoder modules we use locally live under [`scarf/`](./scarf/README.md).

We keep the SCARF dependencies isolated because they are much heavier than the main project and have a different compatibility profile.

Set up the SCARF environment with:

```bash
cd envs/scarf
uv sync
cd ../..
```

This creates `envs/scarf/.venv/`.

## SCARF Weights

For RNA-only embedding, download the Zenodo `model_files.zip` archive and extract only the files we need into `data/reference/scarf/`:

```bash
mkdir -p data/reference/scarf
curl -L "https://zenodo.org/records/17205044/files/model_files.zip?download=1" \
  -o data/reference/scarf/model_files.zip
unzip data/reference/scarf/model_files.zip \
  "model_files/weights/*" \
  "model_files/prior_data/hm_ENSG2token_dict.pickle" \
  -d data/reference/scarf
```

Then copy the extracted files into the paths expected by the local scripts:

```bash
mkdir -p data/reference/scarf/weights data/reference/scarf/prior_data
cp data/reference/scarf/model_files/weights/* data/reference/scarf/weights/
cp data/reference/scarf/model_files/prior_data/hm_ENSG2token_dict.pickle \
  data/reference/scarf/prior_data/
```

For the RNA encoder smoke test, only the `weights/` files are required:

- `weights/config.json`
- `weights/pytorch_model.bin.index.json`
- `weights/pytorch_model-00001-of-00002.bin`
- `weights/pytorch_model-00002-of-00002.bin`

The RNA embedding script also needs:

- `prior_data/hm_ENSG2token_dict.pickle`
- `prior_data/RNA_nonzero_median_10W.hg38.pickle`

Important caveat: the newer `17205044` `model_files.zip` archive does not include `RNA_nonzero_median_10W.hg38.pickle`, even though the raw-count preprocessing path needs it. The checkpoint smoke test can still run with just the weights, but converting raw counts into SCARF-ready RNA inputs still requires that missing median file from another official source.

## SCARF On Google Colab

This repo now includes a Colab-oriented bootstrap path:

- [`scripts/bootstrap_scarf_colab.sh`](./scripts/bootstrap_scarf_colab.sh) installs Colab-compatible SCARF runtime dependencies and downloads the required model assets.
- [`notebooks/scarf_scp3357_colab.ipynb`](./notebooks/scarf_scp3357_colab.ipynb) provides an end-to-end notebook that clones the repo, bootstraps SCARF, builds SCP3357 if needed, and runs the RNA encoder on GPU.

The bootstrap script is intended for a Linux x86_64 Colab GPU runtime. It inspects the live Python, torch, CUDA, and CXX ABI configuration, then resolves matching prebuilt `mamba-ssm` and `causal-conv1d` wheels from the GitHub release assets instead of relying on source builds.

Typical Colab setup from the repository root is:

```bash
bash scripts/bootstrap_scarf_colab.sh
```

Then run the standard embedding command on a GPU runtime:

```bash
python scripts/add_scarf_embeddings.py \
  --input-h5ad /path/to/input.h5ad \
  --output-h5ad /path/to/output_with_scarf.h5ad \
  --device cuda
```

The notebook supports two input paths:

- build a fresh SCP3357 `.h5ad` from a Single Cell Portal bundle stored in Drive or another mounted location
- start from an existing `.h5ad` that already has `layers["counts"]`

Current Colab caveats:

- Use a GPU runtime. CPU runtimes can load the checkpoint weights, but end-to-end SCARF inference still depends on GPU-oriented Mamba/Triton kernels.
- The bootstrap script also streams `RNA_nonzero_median_10W.hg38.pickle` from the older `16956913` SCARF Zenodo bundle, because the newer `17205044` bundle does not include it.
- If Colab changes its base Python or torch runtime and no matching prebuilt wheels are available, the bootstrap script will stop with a clear compatibility error instead of trying to compile the Mamba stack from source.

## SCARF Tests

Run the dedicated SCARF tests from the repository root:

```bash
uv run --project envs/scarf pytest envs/scarf/tests -q
```

The checkpoint smoke test skips automatically until the weights are available under `data/reference/scarf/weights`, or at the path pointed to by `SCARF_CHECKPOINT_DIR`.

## SCARF Embeddings

Use [`scripts/add_scarf_embeddings.py`](./scripts/add_scarf_embeddings.py) from the SCARF environment to add RNA embeddings to an `.h5ad` file:

```bash
uv run --project envs/scarf python scripts/add_scarf_embeddings.py \
  --input-h5ad /path/to/input.h5ad \
  --output-h5ad /path/to/output_with_scarf.h5ad
```

Set `--device cuda` for real SCARF inference. The script now stops with a clear error on CPU runtimes, because the current upstream Mamba2 forward path still requires CUDA for end-to-end embeddings.

The script expects:

- raw counts in `adata.layers["counts"]`
- Ensembl-style gene IDs in `adata.var["gene_ids"]`, or gene symbols that can be backfilled to Ensembl IDs through the cached MyGene lookup used by the script

It writes the embeddings to `adata.obsm["X_scarf"]` by default and stores run metadata in `adata.uns["scarf_embedding"]`.

For SCP3357 specifically, the current conversion keeps gene symbols as `var_names`. The SCARF embedding script backfills `adata.var["gene_ids"]` on the fly, caches the symbol-to-Ensembl mapping in `data/reference/scarf/gene_symbol_to_ensembl.json`, and records per-gene mapping status columns in `adata.var`.

Current runtime caveat: with the upstream SCARF Mamba2 stack and the official checkpoint, end-to-end embedding inference still requires CUDA in this repo. The local test suite validates that the checkpoint weights load on CPU, but the actual forward path depends on GPU-oriented Triton scan kernels.

## Hugging Face Workflow

The root environment includes the Hugging Face Hub CLI as `hf`, so after `uv sync` you can use it with `uv run`.

For this dataset repo, the recommended day-to-day workflow is:

- Use `hf download` to pull the whole dataset snapshot or selected files.
- Use `hf upload` for small updates.
- Use `hf upload-large-folder` for very large dataset syncs.
- Prefer the CLI over `git clone` for routine interaction with the data repo. Use Git/Xet only when you specifically need repository semantics such as branches, history, or direct repo inspection.

### Authentication

The dataset repo is public, so read-only downloads typically do not require login.

If you need to upload data or access protected resources, log in from the command line with:

```bash
uv run hf auth login
uv run hf auth whoami
```

### Download Examples

Download the current dataset snapshot into a local directory:

```bash
uv run hf download pvd232/HopTF --repo-type dataset --local-dir data/hf/HopTF
```

Download a specific file from the dataset repo:

```bash
uv run hf download pvd232/HopTF path/in/repo --repo-type dataset --local-dir data/hf/HopTF
```

### Upload Examples

Upload a file or folder into the dataset repo:

```bash
uv run hf upload pvd232/HopTF ./local/path /path/in/repo --repo-type dataset --commit-message "Update dataset files"
```

For very large uploads, prefer:

```bash
uv run hf upload-large-folder pvd232/HopTF ./local_data_dir --repo-type dataset
```

## Install `uv`

If `uv` is not already installed, one simple option is:

```bash
python3 -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
```

## Set Up The Main Environment

From the repository root:

```bash
uv sync
```

This creates `.venv/` from the root [`uv.lock`](./uv.lock).

Note: the dedicated SCARF environment targets Python `3.12`, matching the upstream project more closely. The root environment stays lighter and is intended for the general repository workflow.

## Run Tests

The test suite lives under [`tests/`](./tests).

Run it with:

```bash
uv run pytest
```

For a quieter success-only run:

```bash
uv run pytest -q
```

## Set Up The AlphaGenome Environment

The AlphaGenome dependencies are intentionally isolated in their own `uv` project:

```bash
cd envs/alphagenome
uv sync
```

This creates `envs/alphagenome/.venv/` from [`envs/alphagenome/uv.lock`](./envs/alphagenome/uv.lock).

Use this environment when working with AlphaGenome-specific scripts or notebooks. Keep using the root environment for general repository work, and `envs/scarf` for the SCARF foundation model.
