"""python -m bagel_space.understanding image.jpg "Describe this image." --layers 0 8 16"""
import argparse
import json
from pathlib import Path

import torch

from .adapter import BagelAdapter
from .capture import Capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("question")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 8, 16])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/understanding"))
    args = parser.parse_args()
    adapter = BagelAdapter()
    adapter.seed(args.seed)
    image = adapter.prepare_understanding(args.image)
    with Capture(adapter.layers, args.layers, "image") as visual, Capture(
        adapter.layers, args.layers, "text",
    ) as text:
        context = adapter.forward_understanding(image, args.question)
    answer = adapter.answer(context, args.max_new_tokens)
    metadata = dict(
        **adapter.metadata(), image=str(args.image.resolve()), question=args.question,
        image_size=list(image.size), layers=args.layers, seed=args.seed, answer=answer,
        alignment=dict(image=visual.alignment, text=text.alignment),
        text_token_ids=[adapter.token_ids["bos_token_id"]]
        + adapter.tokenizer.encode(args.question) + [adapter.token_ids["eos_token_id"]],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "answer.txt").write_text(answer + "\n", encoding="utf-8")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    torch.save(
        dict(hidden_states=dict(image=visual.states, text=text.states), metadata=metadata),
        args.output / "hidden_states.pt",
    )
    print(answer)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
