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
