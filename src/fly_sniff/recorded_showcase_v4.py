from __future__ import annotations

import argparse

from .recorded_showcase_v3 import DEFAULT_E002C, DEFAULT_EVIDENCE_CONFIG, DEFAULT_FC2
from .showcase_v4_runtime import render_v4


def main() -> None:
    parser = argparse.ArgumentParser(description="Render evidence-gated showcase v4")
    parser.add_argument("recording")
    parser.add_argument("--e002c-report", default=str(DEFAULT_E002C))
    parser.add_argument("--fc2-audit", default=str(DEFAULT_FC2))
    parser.add_argument("--evidence-config", default=str(DEFAULT_EVIDENCE_CONFIG))
    parser.add_argument("--output", default="artifacts/showcase/rapid-v4.mp4")
    parser.add_argument("--seconds", type=int, default=16)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    print(
        render_v4(
            args.recording,
            args.output,
            e002c_report=args.e002c_report,
            fc2_audit=args.fc2_audit,
            evidence_config=args.evidence_config,
            seconds=args.seconds,
            fps=args.fps,
        )
    )


if __name__ == "__main__":
    main()
