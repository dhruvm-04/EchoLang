from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import EchoLangPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EchoLang multilingual inference")
    parser.add_argument("--model", required=True, help="Path to echolang bundle pkl")
    parser.add_argument("--text", help="Input query text")
    parser.add_argument("--audio", help="Path to input audio file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.text and not args.audio:
        raise SystemExit("Provide either --text or --audio.")

    pipeline = EchoLangPipeline.load(args.model)

    if args.audio:
        file_path = Path(args.audio)
        audio_bytes = file_path.read_bytes()
        result = pipeline.analyze_audio(audio_bytes=audio_bytes, suffix=file_path.suffix or ".wav")
    else:
        result = pipeline.analyze_text(args.text)

    print(result)


if __name__ == "__main__":
    main()
