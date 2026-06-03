# Method Boundary

PANDA targets exact selected normalization for structured-loss emission sites.

Inputs:

```text
hidden_sites [N,H]
output_weight [C,H]
output_bias [C]
selected_ids [N,S_max]
selected_mask [N,S_max]
selected_adjoints [N,S_max]
```

`C` is output class count / vocabulary size. For each site `i`, PANDA computes
the exact full-`C` normalizer:

```text
logZ_i = logsumexp_v (hidden_i dot output_weight[v] + output_bias[v])
```

and returns selected log-probabilities only for active selected classes in
`S_i`.

Backward consumes compact selected-emission adjoints and produces:

```text
grad_hidden [N,H]
grad_weight [C,H]
grad_bias [C]
```

The PANDA path recomputes logits tile-by-tile and accumulates directly into
these gradients. It does not keep persistent `[N,C]` logits and does not keep
persistent `[N,C]` `grad_logits`.

The dense reference in this repository is intentionally different: it
materializes `[N,C]` logits and uses PyTorch autograd as a parity oracle.

## Standard Blank/Target Case

For standard RNN-T-style selected emissions:

```text
S_i = {target_i, blank}
```

Invalid target sites use an inactive target selected entry and zero target
adjoint. Valid transcript targets must be distinct from `blank_id`.

## Multi-Selected Synthetic Case

The `examples/multiselected_parity_smoke.py` script constructs synthetic
selected sets with `|S_i|=3..5`. This demonstrates selected-set algebra and
compact-adjoint backward parity beyond the two-emission case.

It is not a production multi-blank or TDT implementation.
