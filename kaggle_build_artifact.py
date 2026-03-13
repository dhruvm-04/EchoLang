"""Build a production artifact in Kaggle.

Example:
!python kaggle_build_artifact.py --output /kaggle/working/echolang_bundle.pkl
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.echolang.pipeline import EchoLangPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EchoLang and export a PKL model bundle")
    parser.add_argument(
        "--output",
        default="/kaggle/working/echolang_bundle.pkl",
        help="Destination path for the pickle artifact",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic training",
    )
    parser.add_argument(
        "--no-sbert",
        action="store_true",
        help="Force TF-IDF backend if sentence-transformers is unavailable",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    pipeline, metrics = EchoLangPipeline.train_default(
        random_state=args.seed,
        prefer_sbert=not args.no_sbert,
    )
    pipeline.save(args.output)

    card_path = Path(args.output).with_suffix(".model_card.txt")
    model_card = pipeline.model_card()
    card_path.write_text(str(model_card), encoding="utf-8")

    print("Build complete")
    print(f"Artifact: {args.output}")
    print(f"Model card: {card_path}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
