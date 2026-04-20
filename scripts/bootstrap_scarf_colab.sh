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
  - The script logs each major step with timestamps so notebook output is
    easier to monitor.
  - If the legacy RNA_nonzero_median_10W.hg38.pickle file is unavailable from
    the public SCARF bundles, the embedding script will derive approximate
    per-gene medians from the input dataset instead.
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
SUPPORTED_TORCH_VERSION="2.6.0"
SUPPORTED_TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
STEP_COUNTER=0

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

log_info() {
  echo "[$(timestamp)] $*"
}

log_step() {
  STEP_COUNTER=$((STEP_COUNTER + 1))
  log_info "Step ${STEP_COUNTER}: $*"
}

log_warn() {
  echo "[$(timestamp)] WARNING: $*" >&2
}

mkdir -p "${SCARF_DIR}/weights" "${SCARF_DIR}/prior_data"
log_info "Using SCARF asset directory: ${SCARF_DIR}"

if [[ "${INSTALL_RUNTIME}" -eq 1 ]]; then
  log_step "Inspecting the runtime and selecting Colab-compatible SCARF wheels"
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


def find_asset(repo: str, tag: str, stem: str, py_tag: str, torch_tag: str, abi: str, cuda_candidates: list[str]):
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
        return None

    for cuda_tag in cuda_candidates:
        for asset in matches:
            if f"+{cuda_tag}torch{torch_tag}" in asset["name"]:
                return asset["browser_download_url"], asset["name"]

    return None


def choose_runtime(torch_version_raw: str, abi: str, cuda_version: str | None):
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    torch_tag = normalize_torch_version(torch_version_raw)
    cuda_candidates: list[str] = []
    if cuda_version:
        cuda_major = cuda_version.split(".", 1)[0]
        cuda_candidates.append(f"cu{cuda_major}")
    for fallback in ("cu12", "cu11"):
        if fallback not in cuda_candidates:
            cuda_candidates.append(fallback)

    mamba = find_asset(
        repo="state-spaces/mamba",
        tag="v2.3.1",
        stem="mamba_ssm-",
        py_tag=py_tag,
        torch_tag=torch_tag,
        abi=abi,
        cuda_candidates=cuda_candidates,
    )
    causal = find_asset(
        repo="Dao-AILab/causal-conv1d",
        tag="v1.6.1.post4",
        stem="causal_conv1d-",
        py_tag=py_tag,
        torch_tag=torch_tag,
        abi=abi,
        cuda_candidates=cuda_candidates,
    )
    return py_tag, torch_tag, cuda_candidates, mamba, causal


abi = "TRUE" if bool(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", False)) else "FALSE"
cuda_version = torch.version.cuda
py_tag, torch_tag, cuda_candidates, mamba, causal = choose_runtime(torch.__version__, abi, cuda_version)

needs_torch_reset = mamba is None or causal is None
selected_torch_version = torch.__version__.split("+", 1)[0]
selected_abi = abi

if needs_torch_reset:
    selected_torch_version = "2.6.0"
    selected_abi = "FALSE"
    py_tag, torch_tag, cuda_candidates, mamba, causal = choose_runtime(selected_torch_version, selected_abi, cuda_version)
    if mamba is None or causal is None:
        raise SystemExit(
            f"No compatible SCARF wheels found for python={py_tag}. "
            f"Tried live torch={torch.__version__} abi={abi} and fallback torch={selected_torch_version} abi={selected_abi}."
        )

mamba_url, mamba_name = mamba
causal_url, causal_name = causal

print(f"PYTHON_VERSION={json.dumps(f'{sys.version_info.major}.{sys.version_info.minor}')}")
print(f"LIVE_TORCH_VERSION={json.dumps(torch.__version__.split('+', 1)[0])}")
print(f"SELECTED_TORCH_VERSION={json.dumps(selected_torch_version)}")
print(f"LIVE_TORCH_CXX11ABI={json.dumps(abi)}")
print(f"SELECTED_TORCH_CXX11ABI={json.dumps(selected_abi)}")
print(f"TORCH_CUDA={json.dumps(cuda_version or '')}")
print(f"GPU_RUNTIME={json.dumps('1' if torch.cuda.is_available() else '0')}")
print(f"RESET_TORCH={json.dumps('1' if needs_torch_reset else '0')}")
print(f"MAMBA_URL={json.dumps(mamba_url)}")
print(f"MAMBA_WHEEL={json.dumps(mamba_name)}")
print(f"CAUSAL_CONV_URL={json.dumps(causal_url)}")
print(f"CAUSAL_CONV_WHEEL={json.dumps(causal_name)}")
PY
  )"

  log_info "Python runtime: ${PYTHON_VERSION}"
  log_info "Live torch runtime: ${LIVE_TORCH_VERSION} (CUDA ${TORCH_CUDA:-unknown}, cxx11abi=${LIVE_TORCH_CXX11ABI})"
  if [[ "${RESET_TORCH}" == "1" ]]; then
    log_step "Resetting torch to ${SELECTED_TORCH_VERSION} for SCARF wheel compatibility"
    python -m pip install --quiet --upgrade pip
    python -m pip install --quiet --force-reinstall \
      --index-url "${SUPPORTED_TORCH_INDEX_URL}" \
      "torch==${SUPPORTED_TORCH_VERSION}" \
      "torchvision==0.21.0" \
      "torchaudio==2.6.0"
  else
    log_step "Keeping the existing torch runtime and upgrading pip"
    python -m pip install --quiet --upgrade pip
  fi
  log_info "Selected torch runtime: ${SELECTED_TORCH_VERSION} (cxx11abi=${SELECTED_TORCH_CXX11ABI})"
  log_step "Installing Python dependencies, ${CAUSAL_CONV_WHEEL}, and ${MAMBA_WHEEL}"

  python -m pip install --quiet \
    "anndata>=0.9,<0.12" \
    "huggingface-hub>=0.25,<1.0" \
    "pandas" \
    "scipy" \
    "tqdm" \
    "transformers==4.46.3"
  python -m pip install --quiet "${CAUSAL_CONV_URL}" "${MAMBA_URL}"

  if [[ "${GPU_RUNTIME}" != "1" ]]; then
    log_warn "No CUDA runtime detected. Weight loading should work, but end-to-end SCARF embeddings still need a GPU."
  fi
else
  log_info "Skipping Python package installation."
fi

if [[ "${DOWNLOAD_MODEL}" -eq 1 ]]; then
  if [[ "${FORCE}" -eq 1 ]] || [[ ! -s "${SCARF_DIR}/weights/config.json" ]] || [[ ! -s "${SCARF_DIR}/prior_data/hm_ENSG2token_dict.pickle" ]]; then
    log_step "Downloading SCARF model_files.zip (~6.4 GB)"
    wget -c -O "${SCARF_DIR}/model_files.zip" "${MODEL_ZIP_URL}"
    log_step "Extracting SCARF weights and token dictionary"
    UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -q -o \
      "${SCARF_DIR}/model_files.zip" \
      "model_files/weights/*" \
      "model_files/prior_data/hm_ENSG2token_dict.pickle" \
      -d "${SCARF_DIR}"
    cp -f "${SCARF_DIR}/model_files/weights/"* "${SCARF_DIR}/weights/"
    cp -f "${SCARF_DIR}/model_files/prior_data/hm_ENSG2token_dict.pickle" "${SCARF_DIR}/prior_data/"
    rm -rf "${SCARF_DIR}/model_files"
  else
    log_info "Skipping SCARF weight download; expected files already exist."
  fi
else
  log_info "Skipping SCARF weight download by request."
fi

if [[ "${DOWNLOAD_MEDIAN}" -eq 1 ]]; then
  MEDIAN_TARGET="${SCARF_DIR}/prior_data/RNA_nonzero_median_10W.hg38.pickle"
  if [[ "${FORCE}" -eq 1 ]] || [[ ! -s "${MEDIAN_TARGET}" ]]; then
    log_step "Attempting to recover RNA_nonzero_median_10W.hg38.pickle from the legacy SCARF bundle"
    if LEGACY_MODEL_TAR_URL="${LEGACY_MODEL_TAR_URL}" MEDIAN_TARGET="${MEDIAN_TARGET}" python - <<'PY'
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
    then
      log_info "Recovered the legacy RNA median file."
    else
      log_warn "Could not recover RNA_nonzero_median_10W.hg38.pickle from the public SCARF bundles."
      log_warn "SCARF embedding will fall back to deriving per-gene nonzero medians from the input dataset."
    fi
  else
    log_info "Skipping legacy median download; file already exists."
  fi
else
  log_info "Skipping legacy median download by request."
fi

log_step "Bootstrap complete"
log_info "Assets directory: ${SCARF_DIR}"
