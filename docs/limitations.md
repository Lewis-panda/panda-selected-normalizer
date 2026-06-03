# Limitations

- This is a research prototype package, not a production/default backend.
- The default implementation is a readable PyTorch tiled/manual-backward path,
  not an optimized Triton kernel.
- The package does not run training or decode.
- The package does not report ASR quality or WER/CER.
- The package does not claim full-training speedup.
- The package does not claim multi-hardware generality.
- The package does not include k2/icefall as base dependencies.
- The multi-selected smoke is synthetic parity only; it is not production
  multi-blank or TDT support.
- The dense reference intentionally materializes full logits and uses autograd;
  it is a small-shape oracle, not the memory-bounded path.
