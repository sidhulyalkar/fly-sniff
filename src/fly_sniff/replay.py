from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from .graph import GraphBundle
from .runtime import ConnectomeRuntime


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frames(path: str | Path) -> list[dict]:
    frames: list[dict] = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if "inputs" not in payload or not isinstance(payload["inputs"], dict):
                raise ValueError(f"line {line_number} must contain an inputs object")
            frames.append(payload)
    if not frames:
        raise ValueError("stimulus replay contains no frames")
    return frames


def replay_frames(
    bundle: GraphBundle,
    frames: Iterable[dict],
    *,
    readouts: Iterable[str],
    require_qualified: bool = True,
    leak: float = 0.82,
    gain: float = 1.6,
) -> list[dict]:
    runtime = ConnectomeRuntime(
        bundle,
        leak=leak,
        gain=gain,
        require_qualified=require_qualified,
    )
    runtime.reset()
    requested = tuple(readouts)
    trace: list[dict] = []
    for index, frame in enumerate(frames):
        snapshot = runtime.step(frame["inputs"], readouts=requested)
        trace.append(
            {
                "frame": index,
                "source_t": frame.get("t", index),
                "inputs": snapshot.inputs,
                "readouts": snapshot.readouts,
                "activity_mean": snapshot.activity_mean,
                "activity_max": snapshot.activity_max,
            }
        )
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay recorded adapter channels through a connectome graph deterministically"
    )
    parser.add_argument("graph", help="GraphBundle directory")
    parser.add_argument("stimulus", help="JSONL frames: {'t': ..., 'inputs': {'role': value}}")
    parser.add_argument("--readout", action="append", default=[], help="role to record; repeatable")
    parser.add_argument("--output", default="artifacts/replay-trace.jsonl")
    parser.add_argument(
        "--allow-candidate",
        action="store_true",
        help="development only: run an unqualified graph; output remains non-evidence",
    )
    parser.add_argument("--leak", type=float, default=0.82)
    parser.add_argument("--gain", type=float, default=1.6)
    args = parser.parse_args()

    graph_root = Path(args.graph)
    stimulus_path = Path(args.stimulus)
    output_path = Path(args.output)
    bundle = GraphBundle.load(graph_root)
    frames = load_frames(stimulus_path)
    trace = replay_frames(
        bundle,
        frames,
        readouts=args.readout,
        require_qualified=not args.allow_candidate,
        leak=args.leak,
        gain=args.gain,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as handle:
        for row in trace:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    receipt = {
        "claim_status": "development-candidate" if args.allow_candidate else "qualified-graph-replay",
        "frames": len(trace),
        "readouts": list(args.readout),
        "stimulus": str(stimulus_path),
        "stimulus_sha256": _sha256(stimulus_path),
        "graph_manifest": bundle.manifest,
        "dynamics": {"model": "rate-v0", "leak": args.leak, "gain": args.gain},
        "trace": str(output_path),
        "trace_sha256": _sha256(output_path),
    }
    receipt_path = output_path.with_suffix(output_path.suffix + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
