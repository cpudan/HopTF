#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Bootstrap a Google Colab runtime for SCARF RNA embeddings.

Usage:
  bash scripts/bootstrap_scarf_colab.sh [options]

Options:
  --scarf-dir DIR            Override the SCARF asset directory.
  --skip-install             Skip Python package installation.
  --skip-model-download      Skip downloading the SCARF weight bundle.
  --skip-median-download     Skip downloading the legacy RNA median file.
  --force                    Redownload or overwrite local assets.
  -h, --help                 Show this help text.

Notes:
  - This script is intended for a Colab GPU runtime.
  - It selects matching prebuilt mamba-ssm and causal-conv1d wheels from
    GitHub release assets based on the live Python / torch / CUDA / CXX ABI.
  - The current SCARF raw-count preprocessing path also needs the legacy
    RNA_nonzero_median_10W.hg38.pickle file, which is streamed from the older
    SCARF Zenodo bundle.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCARF_DIR="${ROOT_DIR}/data/reference/scarf"
INSTALL_RUNTIME=1
DOWNLOAD_MODEL=1
DOWNLOAD_MEDIAN=1
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scarf-dir)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --scarf-dir" >&2
        exit 1
      fi
      SCARF_DIR="$2"
      shift 2
      ;;
    --skip-install)
      INSTALL_RUNTIME=0
      shift
      ;;
    --skip-model-download)
      DOWNLOAD_MODEL=0
      shift
      ;;
    --skip-median-download)
      DOWNLOAD_MEDIAN=0
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

MODEL_ZIP_URL="https://zenodo.org/api/records/17205044/files/model_files.zip/content"
LEGACY_MODEL_TAR_URL="https://zenodo.org/api/records/16956913/files/model_files.tar.gz/content"

mkdir -p "${SCARF_DIR}/weights" "${SCARF_DIR}/prior_data"

if [[ "${INSTALL_RUNTIME}" -eq 1 ]]; then
  echo "Inspecting runtime and selecting Colab-compatible SCARF wheels..."
  eval "$(
    python - <<'PY'
import json
import sys
import urllib.request

try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch must already be available in this runtime.") from exc


def normalize_torch_version(raw: str) -> str:
    base = raw.split("+", 1)[0]
    parts = base.split(".")
    return ".".join(parts[:2])


def select_asset(repo: str, tag: str, stem: str, py_tag: str, torch_tag: str, abi: str, cuda_candidates: list[str]):
    url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    with urllib.request.urlopen(url) as response:
        assets = json.load(response)["assets"]

    matches = []
    for asset in assets:
        name = asset["name"]
        if not name.startswith(stem):
            continue
        if f"torch{torch_tag}" not in name:
            continue
        if f"cxx11abi{abi}" not in name:
            continue
        if f"{py_tag}-{py_tag}" not in name:
            continue
        if not name.endswith("linux_x86_64.whl"):
            continue
        matches.append(asset)

    if not matches:
        raise SystemExit(
            f"No prebuilt {stem} wheel matches python={py_tag}, torch={torch_tag}, cxx11abi={abi}."
        )

    for cuda_tag in cuda_candidates:
        for asset in matches:
            if f"+{cuda_tag}torch{torch_tag}" in asset["name"]:
                return asset["browser_download_url"], asset["name"]

    candidate_names = ", ".join(asset["name"] for asset in matches)
    raise SystemExit(
        f"No compatible CUDA-tagged {stem} wheel found for python={py_tag}, torch={torch_tag}, "
        f"cxx11abi={abi}. Candidates: {candidate_names}"
    )


py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
torch_tag = normalize_torch_version(torch.__version__)
abi = "TRUE" if bool(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", False)) else "FALSE"
cuda_version = torch.version.cuda
cuda_candidates: list[str] = []
if cuda_version:
    cuda_major = cuda_version.split(".", 1)[0]
    cuda_candidates.append(f"cu{cuda_major}")
for fallback in ("cu12", "cu11"):
    if fallback not in cuda_candidates:
        cuda_candidates.append(fallback)

mamba_url, mamba_name = select_asset(
    repo="state-spaces/mamba",
    tag="v2.3.1",
    stem="mamba_ssm-",
    py_tag=py_tag,
    torch_tag=torch_tag,
    abi=abi,
    cuda_candidates=cuda_candidates,
)
causal_url, causal_name = select_asset(
    repo="Dao-AILab/causal-conv1d",
    tag="v1.6.1.post4",
    stem="causal_conv1d-",
    py_tag=py_tag,
    torch_tag=torch_tag,
    abi=abi,
    cuda_candidates=cuda_candidates,
)

print(f"PYTHON_VERSION={json.dumps(f'{sys.version_info.major}.{sys.version_info.minor}')}")
print(f"TORCH_VERSION={json.dumps(torch.__version__.split('+', 1)[0])}")
print(f"TORCH_CUDA={json.dumps(cuda_version or '')}")
print(f"GPU_RUNTIME={json.dumps('1' if torch.cuda.is_available() else '0')}")
print(f"MAMBA_URL={json.dumps(mamba_url)}")
print(f"MAMBA_WHEEL={json.dumps(mamba_name)}")
print(f"CAUSAL_CONV_URL={json.dumps(causal_url)}")
print(f"CAUSAL_CONV_WHEEL={json.dumps(causal_name)}")
PY
  )"

  echo "Python runtime: ${PYTHON_VERSION}"
  echo "Torch runtime: ${TORCH_VERSION} (CUDA ${TORCH_CUDA:-unknown})"
  echo "Installing ${CAUSAL_CONV_WHEEL}"
  echo "Installing ${MAMBA_WHEEL}"

  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet \
    "anndata>=0.9,<0.12" \
    "huggingface-hub>=0.25,<1.0" \
    "pandas" \
    "scipy" \
    "tqdm" \
    "transformers==4.46.3"
  python -m pip install --quiet "${CAUSAL_CONV_URL}" "${MAMBA_URL}"

  if [[ "${GPU_RUNTIME}" != "1" ]]; then
    echo "WARNING: No CUDA runtime detected. Weight loading should work, but end-to-end SCARF embeddings still need a GPU." >&2
  fi
fi

if [[ "${DOWNLOAD_MODEL}" -eq 1 ]]; then
  if [[ "${FORCE}" -eq 1 ]] || [[ ! -s "${SCARF_DIR}/weights/config.json" ]] || [[ ! -s "${SCARF_DIR}/prior_data/hm_ENSG2token_dict.pickle" ]]; then
    echo "Downloading SCARF model_files.zip (~6.4 GB)..."
    wget -c -O "${SCARF_DIR}/model_files.zip" "${MODEL_ZIP_URL}"
    echo "Extracting SCARF weights and token dictionary..."
    UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -q -o \
      "${SCARF_DIR}/model_files.zip" \
      "model_files/weights/*" \
      "model_files/prior_data/hm_ENSG2token_dict.pickle" \
      -d "${SCARF_DIR}"
    cp -f "${SCARF_DIR}/model_files/weights/"* "${SCARF_DIR}/weights/"
    cp -f "${SCARF_DIR}/model_files/prior_data/hm_ENSG2token_dict.pickle" "${SCARF_DIR}/prior_data/"
    rm -rf "${SCARF_DIR}/model_files"
  else
    echo "Skipping SCARF weight download; expected files already exist."
  fi
fi

if [[ "${DOWNLOAD_MEDIAN}" -eq 1 ]]; then
  MEDIAN_TARGET="${SCARF_DIR}/prior_data/RNA_nonzero_median_10W.hg38.pickle"
  if [[ "${FORCE}" -eq 1 ]] || [[ ! -s "${MEDIAN_TARGET}" ]]; then
    echo "Streaming RNA_nonzero_median_10W.hg38.pickle from the legacy SCARF bundle..."
    LEGACY_MODEL_TAR_URL="${LEGACY_MODEL_TAR_URL}" MEDIAN_TARGET="${MEDIAN_TARGET}" python - <<'PY'
import os
import tarfile
import urllib.request

url = os.environ["LEGACY_MODEL_TAR_URL"]
target = os.environ["MEDIAN_TARGET"]
needle = "prior_data/RNA_nonzero_median_10W.hg38.pickle"
tmp_target = f"{target}.tmp"

os.makedirs(os.path.dirname(target), exist_ok=True)

try:
    with urllib.request.urlopen(url) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as tar:
            for member in tar:
                if member.name == needle or member.name.endswith("/" + needle):
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        raise RuntimeError(f"Could not extract {needle} from {url}")
                    with open(tmp_target, "wb") as handle:
                        while True:
                            chunk = extracted.read(1024 * 1024)
                            if not chunk:
                                break
                            handle.write(chunk)
                    os.replace(tmp_target, target)
                    print(f"Wrote {target}")
                    break
            else:
                raise RuntimeError(f"Did not find {needle} in {url}")
finally:
    if os.path.exists(tmp_target):
        os.remove(tmp_target)
PY
  else
    echo "Skipping legacy median download; file already exists."
  fi
fi

echo "SCARF Colab bootstrap complete."
echo "Assets directory: ${SCARF_DIR}"
