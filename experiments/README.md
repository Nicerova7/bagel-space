# Experiments

Use [the custom notebook](../colab-custom-steering.ipynb) for one pair and the reusable functions.
Use [the counting notebook](counting/colab-counting.ipynb) for dataset builds and image steering.
`custom_utils.py` reuses the BAGEL adapter and capture module without changing upstream code.

Counting steps 3-6 prepare pairs, capture understanding states, and average pair differences per layer.
Token and layer reducers accept `mean`, `average`, `sum`, `none`, or a callable.
RepE and ITI have reserved sections only.

Application tokens, layers, and visual iterations accept `first`, `last`, `all`, integers, lists, or slices.
Text edits run once on prompt content before caching; visual edits run on conditional VAE tokens.
Visual iterations exclude CFG reference forwards. Special tokens are never application targets.

- `add`: `h + alpha*v`.
- `component`: `h + alpha*(h @ unit_v)*unit_v`; `alpha=-1` removes the projection.
- Custom operation: `operation(hidden, vector, *, alpha)` returns edited states.

Extraction includes BOS and no EOS; `right_pad` appends spaces before last-token selection.
Build each layer's vector before selecting it for application. Local layer indices are zero-based.
