"""python -m bagel_space.generation "A photograph of a dog." --layers 0 8 16"""
import argparse
import json
from pathlib import Path

import torch

from .adapter import BagelAdapter
from .capture import Capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 8, 16])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/generation"))
    args = parser.parse_args()
    adapter = BagelAdapter()
    with Capture(adapter.layers, args.layers, "generation") as captured:
        image = adapter.generate(args.prompt, args.steps, args.guidance, args.seed)
    metadata = dict(
        **adapter.metadata(), prompt=args.prompt, resolution=1024, layers=args.layers,
        steps=args.steps, guidance=args.guidance, seed=args.seed,
        image_guidance=1.0, cfg_interval=[0.4, 1.0], timestep_shift=3.0,
        cfg_renorm_min=0.0, cfg_renorm_type="global", taylorseer=False,
        evaluation="first conditional flow evaluation", alignment=captured.alignment,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    image.save(args.output / "image.png")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    torch.save(dict(hidden_states=captured.states, metadata=metadata), args.output / "hidden_states.pt")
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
