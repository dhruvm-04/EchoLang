from __future__ import annotations

import argparse

from .pipeline import EchoLangPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EchoLang inference from a PKL bundle")
    parser.add_argument("--model", required=True, help="Path to echolang bundle pkl")
    parser.add_argument("--text", required=True, help="Input query text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = EchoLangPipeline.load(args.model)
    result = pipeline.analyze_text(args.text)
    print(result)


if __name__ == "__main__":
    main()
