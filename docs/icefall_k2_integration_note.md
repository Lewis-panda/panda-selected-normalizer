# icefall / k2 Integration Note

This package does not include icefall or k2 as base dependencies.

An integration should place PANDA at the selected-normalizer boundary:

```text
cached or retained hidden sites [N,H]
output head weight/bias [C,H], [C]
selected emission ids and masks
compact selected-emission adjoints from the DP/loss side
  -> selected log-probs, logZ, output-head gradients
```

PANDA replaces the operation that would otherwise materialize full output logits
for retained/full sites and then backpropagate through a full `grad_logits`
tensor.

The integration must not infer recipe semantics. The ASR stack remains
responsible for:

- tokenizer and output-head class count;
- blank id;
- retained-site order and masks;
- compact DP / k2 loss semantics;
- invalid target zero rules;
- optimizer and training loop;
- decode and WER/CER evaluation.

The current package should be treated as a research prototype and smoke-test
surface, not as a production/default backend. A real icefall/k2 integration
needs separate fixture parity, shape gates, memory accounting, and ASR-owner
validation.

## Minimal Call Pattern

For standard blank/target transducer sites, the package boundary is exercised
like this (shapes only; recipe semantics stay outside the package):

```python
import torch
from panda import build_standard_blank_target_set, selected_forward_backward

# hidden  [N, H]  joiner hidden states at consumed (full or retained) sites
# weight  [C, H]  output head weight; bias [C] output head bias
# target_ids [N] transcript-target class per site, -1 marks invalid sites
# d_target, d_blank [N] compact DP adjoints produced by the loss/DP side

selected_ids, selected_mask = build_standard_blank_target_set(
    target_ids, blank_id, num_classes=weight.shape[0]
)
selected_adjoints = torch.stack([d_target, d_blank], dim=1)

out = selected_forward_backward(
    hidden, weight, bias,
    selected_ids, selected_mask, selected_adjoints,
    tile_c=1024,
)
# out.selected_logp [N, 2], out.logZ [N]
# out.grad_hidden [N, H], out.grad_weight [C, H], out.grad_bias [C]
```

Invalid target sites (`target_ids[i] < 0`) keep the blank edge active, omit the
target edge via `selected_mask`, and must receive zero target adjoint.

## Validated k2 / PyTorch Environment Combination

k2 wheels are ABI-coupled to a specific PyTorch/CUDA build. One combination has
passed the recipe-native ABI/import preflight (torch, k2, kaldifeat, icefall,
lhotse, sentencepiece imports plus CPU/CUDA tensor smoke) in this project:

```text
python      3.10
torch       2.9.1+cu128 (CUDA 12.8)
k2          1.24.4.dev20251118+cuda12.8.torch2.9.1
kaldifeat   1.24
```

A known-bad combination, for contrast: the same k2 build fails import against
torch `2.10.0+cu130` with a torch/k2 ABI mismatch. If your k2 import fails,
match the k2 wheel's `torchX.Y.Z+cudaA.B` suffix to the installed torch build
exactly before debugging anything else.

This is one validated same-machine combination, not a multi-platform support
matrix.
