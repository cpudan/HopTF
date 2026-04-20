#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Convenience wrapper for bootstrapping SCARF inside Google Colab.

This wrapper delegates to:
  bash scripts/bootstrap_scarf.sh [options]

It exists so the Colab notebook can keep a stable entrypoint while the main
bootstrap logic stays generic for other Linux GPU environments.
EOF
fi

exec bash "${ROOT_DIR}/scripts/bootstrap_scarf.sh" "$@"
