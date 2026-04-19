This directory contains the minimal encoder modules copied from the official SCARF repository:

- upstream: <https://github.com/cbmi-group/scarf>
- upstream commit used for this copy: `1b51ea8f9e861308b103d5dc70df20cf785626ce`

The original SCARF model architecture, training code, and project structure belong to the upstream authors. This local copy exists only so this repository can depend on a small, pinned, auditable subset of the encoder code without requiring a full Git submodule checkout.

Local adjustments in this repo are intentionally small:

- make the value encoder device-agnostic instead of hard-coding CUDA
- remove repeated model-construction print spam so tests and scripts stay readable
