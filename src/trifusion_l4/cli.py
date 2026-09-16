"""Command-line entry points for an installed TriFusion-FP kernel package."""

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform

from . import __version__


def parser():
    result = argparse.ArgumentParser(prog="trifusion-l4")
    result.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("info", help="Print installed versions without initializing CUDA")
    infer = commands.add_parser("predict", help="Run offline inference with your local checkpoint")
    infer.add_argument("--weights", type=Path, required=True)
    infer.add_argument(
        "--root", type=Path, required=True, help="Directory containing data/ or open/"
    )
    infer.add_argument("--graphs", action=argparse.BooleanOptionalAction, default=True)
    infer.add_argument("--padded-seconds", type=int, default=64)
    infer.add_argument("--max-batch", type=int, default=16)
    infer.add_argument("--workers", type=int, default=4)
    return result


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    if args.command == "info":
        versions = {}
        for name in ("torch", "triton", "numpy", "scipy", "soundfile", "einops"):
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = None
        print(
            json.dumps(
                {
                    "version": __version__,
                    "python": platform.python_version(),
                    "platform": platform.system(),
                    "dependencies": versions,
                },
                indent=2,
            )
        )
        return 0
    if not args.weights.is_file():
        command.error(f"Checkpoint does not exist: {args.weights}")
    if not args.root.is_dir():
        command.error(f"Input root does not exist: {args.root}")
    if min(args.padded_seconds, args.max_batch, args.workers) < 1:
        command.error("Batch limits and worker count must be positive")
    from .submission import run

    run(
        args.root.resolve(),
        args.weights.resolve(),
        graphs=args.graphs,
        padded_seconds=args.padded_seconds,
        max_batch=args.max_batch,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
