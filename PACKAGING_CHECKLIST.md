# PANDA GitHub Packaging Checklist

Date: 2026-06-04

## Included Files

- `panda/`: selected-normalizer implementation, dense small-shape reference,
  and optional Triton boundary placeholder.
- `examples/`: CPU/CUDA smoke scripts for standard blank/target and synthetic
  multi-selected parity.
- `tests/`: parity and input-contract tests.
- `scripts/run_smoke.sh`: local smoke wrapper.
- `docs/`: method boundary, limitations, and icefall/k2 integration note.
- `requirements.txt`, `environment.yml`, `pyproject.toml`: package and smoke
  environment definitions.
- `.github/workflows/smoke.yml`: CPU-only GitHub Actions smoke/test workflow.
- `LICENSE`: Apache License 2.0.

## Smoke Commands

```bash
scripts/run_smoke.sh
python examples/minimal_selected_normalizer_smoke.py
python examples/multiselected_parity_smoke.py
python tests/test_selected_normalizer_parity.py
python -m pytest tests -q
```

## Verification Status

- Large artifacts/paper PDFs/checkpoints: not present in this minimal package.
- Legacy method-name leakage: not present in package files as of this checklist
  pass.
- Workspace-specific role names and review-response wording: removed or not
  present in package files as of this checklist pass.
- Base environment: `torch` and `pytest`; k2/icefall are not base dependencies.
- License: Apache License 2.0.
- Optional Triton extra: documented as a non-production placeholder boundary.
- `scripts/run_smoke.sh`: pass in the existing torch environment; because
  `pytest` was absent there, the wrapper used its direct Python test fallback.
- `python examples/minimal_selected_normalizer_smoke.py`: pass in the existing
  torch environment.
- `python examples/multiselected_parity_smoke.py`: pass in the existing torch
  environment.
- `python tests/test_selected_normalizer_parity.py`: pass in the existing torch
  environment.
- `python -m pytest tests -q`: blocked only in the existing local environment
  because `pytest` is not installed; `requirements.txt`, `environment.yml`, and
  `.[test]` include `pytest` for the package smoke/test environment.
- Fresh venv validation on 2026-06-04:
  - `python3 -m venv .venv`;
  - `pip install -r requirements.txt`;
  - resolved `torch 2.12.0+cu130`, `numpy 2.4.6`, `pytest 9.0.3`;
  - `scripts/run_smoke.sh`: pass;
  - `python -m pytest tests -q`: pass;
  - CPU smoke only; no training/decode/benchmark.
- The fresh validation `.venv`, `.pytest_cache`, and `__pycache__` directories
  were removed after validation.  They are also covered by `.gitignore`.

## Known Limitations

- Research prototype only; not a production/default ASR backend.
- No ASR training, decode, WER/CER, ASR quality, full-training speedup, or
  multi-hardware claim is made by this package.
- The examples use random tensors and small-shape parity against a dense oracle.
- The dense oracle intentionally materializes full logits and uses autograd; it
  is not the memory-bounded PANDA path.
- The package does not include k2/icefall integration code.
- The package does not include packaged Triton kernels.

## Intentionally Excluded

- Research workspace outputs, benchmark CSVs/JSONs, checkpoints, recogs, paper
  PDFs, review-response material, and workspace logs.
- ASR recipes, training loops, decode scripts, full C-sweep benchmark harnesses,
  and second-hardware run records.
