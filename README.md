# bagel-space

Run official BAGEL understanding and generation; capture selected decoder-block states.
Start with [colab.ipynb](colab.ipynb) on one **A100 80 GB**, BF16, without quantization.

```text
bagel_space/   adapter.py, capture.py, generation.py, understanding.py
inputs/       local images
outputs/      images, answers, hidden states, metadata
experiments/  future experiments
```

Understanding saves image and question prefills separately; answer decoding runs afterward.
Generation saves the first conditional flow evaluation, including image boundary tokens.
States are CPU tensors shaped `[tokens, hidden]`, captured before final normalization.
Metadata identifies token positions and pinned model/code revisions. KV cache stays enabled.

With the official BAGEL checkout and this project on `PYTHONPATH`:

```bash
python -m bagel_space.understanding inputs/images/example.jpg "Describe this image." --layers 0 8 16
python -m bagel_space.generation "One yellow rubber duck." --layers 0 8 16
```

Generation defaults follow Cross-Branch Appendix A.1: 1024px, 50 timesteps, CFG 1.0, shift 3.0, seed 42.
GPU inference remains unverified. This project currently contains no steering methods.

Sources: [paper](https://arxiv.org/abs/2505.14683), [official code](https://github.com/ByteDance-Seed/Bagel), [checkpoint](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT).
