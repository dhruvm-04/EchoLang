"""Kaggle build script.

Usage in Kaggle notebook cell:
!python kaggle_build_artifact.py --output /kaggle/working/echolang_bundle.pkl
"""

from __future__ import annotations

import argparse
import os

from src.echolang.pipeline import EchoLangPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EchoLang and export PKL bundle")
    parser.add_argument(
        "--output",
        default="/kaggle/working/echolang_bundle.pkl",
        help="Destination path for the pickle artifact",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    pipeline, metrics = EchoLangPipeline.train_default()
    pipeline.save(args.output)

    print("Build complete")
    print(f"Artifact: {args.output}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
