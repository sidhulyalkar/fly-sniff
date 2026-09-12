from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ANNOTATIONS_PATH = Path("data/raw/body-annotations-male-cns-v1.0.feather")
WEIGHTS_PATH = Path("data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather")
REQUIRED_SCRIPTS = (
    "fly-sniff-trace",
    "fly-sniff-audit-trace",
    "fly-sniff-science-handoff",
)


def _probe_scientific_python() -> dict[str, Any]:
    code = (
        "import json, numpy, scipy; "
        "print(json.dumps({'numpy': numpy.__version__, 'scipy': scipy.__version__}))"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        stderr = getattr(exc, "stderr", None)
        return {
            "ok": False,
            "numpy": None,
            "scipy": None,
            "error": (stderr or str(exc)).strip()[:2000],
        }
    payload = json.loads(completed.stdout.strip())
    return {
        "ok": True,
        "numpy": str(payload["numpy"]),
        "scipy": str(payload["scipy"]),
        "error": None,
    }


def _resolve_console_script(name: str, executable_dir: Path) -> str | None:
    """Resolve a project script, preferring the active interpreter's environment."""
    sibling = executable_dir / name
    if sibling.exists() and os.access(sibling, os.X_OK):
        return str(sibling)
    return shutil.which(name)


def diagnose(
    *,
    root: str | Path = ".",
    environ: Mapping[str, str] | None = None,
    ffmpeg_path: str | None = None,
) -> dict[str, Any]:
    root = Path(root)
    env = os.environ if environ is None else environ
    ffmpeg = shutil.which("ffmpeg") if ffmpeg_path is None else ffmpeg_path
    annotations = root / ANNOTATIONS_PATH
    weights = root / WEIGHTS_PATH
    python_ok = sys.version_info >= (3, 11)
    token_present = bool(env.get("NEUPRINT_TOKEN"))
    virtual_env = sys.prefix != sys.base_prefix
    science = _probe_scientific_python()

    executable_dir = Path(sys.executable).resolve().parent
    scripts: dict[str, Any] = {}
    scripts_same_environment = True
    for name in REQUIRED_SCRIPTS:
        resolved = _resolve_console_script(name, executable_dir)
        same_environment = bool(
            resolved and Path(resolved).resolve().parent == executable_dir
        )
        scripts[name] = {
            "path": resolved,
            "present": bool(resolved),
            "same_environment_as_python": same_environment,
        }
        scripts_same_environment &= same_environment

    return {
        "python": {
            "version": ".".join(str(x) for x in sys.version_info[:3]),
            "executable": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "virtual_env": virtual_env,
            "ok": python_ok,
            "required": ">=3.11",
        },
        "scientific_python": science,
        "console_scripts": {
            "all_same_environment": scripts_same_environment,
            "items": scripts,
        },
        "ffmpeg": {"path": ffmpeg, "ok": bool(ffmpeg)},
        "neuprint_token": {"present": token_present},
        "data": {
            "annotations": {"path": str(annotations), "present": annotations.exists()},
            "weights": {"path": str(weights), "present": weights.exists()},
        },
        "ready": {
            "gif_demo": python_ok and science["ok"],
            "mp4_showcase": python_ok and science["ok"] and bool(ffmpeg),
            "live_malecns": python_ok and science["ok"] and token_present,
            "offline_full_graph_trace": (
                python_ok
                and science["ok"]
                and annotations.exists()
                and weights.exists()
                and scripts_same_environment
            ),
        },
    }


def _print_human(report: dict[str, Any]) -> None:
    yes = "READY"
    no = "MISSING"
    python = report["python"]
    science = report["scientific_python"]
    print(
        f"Python {python['version']} ({yes if python['ok'] else no}; requires >=3.11)\n"
        f"  executable: {python['executable']}\n"
        f"  virtual env: {'YES' if python['virtual_env'] else 'NO'}"
    )
    if science["ok"]:
        print(f"scientific Python: READY (NumPy {science['numpy']} • SciPy {science['scipy']})")
    else:
        print("scientific Python: BROKEN")
        if science["error"]:
            print(f"  {science['error']}")
    print("console scripts:")
    for name, info in report["console_scripts"]["items"].items():
        status = "READY" if info["same_environment_as_python"] else "WRONG/MISSING"
        print(f"  {name}: {status} ({info['path'] or 'not found'})")
    print(f"ffmpeg: {report['ffmpeg']['path'] or no}")
    print(f"NEUPRINT_TOKEN: {yes if report['neuprint_token']['present'] else no}")
    print(
        "annotations: "
        f"{yes if report['data']['annotations']['present'] else no} "
        f"({report['data']['annotations']['path']})"
    )
    print(
        "full weights: "
        f"{yes if report['data']['weights']['present'] else no} "
        f"({report['data']['weights']['path']})"
    )
    print("\nCapabilities")
    for name, ready in report["ready"].items():
        print(f"  {name}: {yes if ready else no}")

    if not python["virtual_env"]:
        print(
            "\nWARNING: not running inside a virtual environment. "
            "Use `.venv/bin/python -m ...` or activate `.venv` before scientific runs."
        )
    if not science["ok"]:
        print(
            "\nScientific Python import failed. Recreate `.venv` and install this project "
            "with the pinned NumPy 1.26 ABI before running tests or MaleCNS analysis."
        )
    if not report["console_scripts"]["all_same_environment"]:
        print(
            "\nConsole scripts are missing or come from a different environment. "
            "Run `.venv/bin/python -m pip install -e '.[malecns,dev]'` and use "
            "`.venv/bin/python -m pytest` to avoid a global pytest executable."
        )
    if not report["ready"]["mp4_showcase"]:
        print("\nInstall ffmpeg for MP4 output, or render GIF immediately.")
    if not report["data"]["annotations"]["present"]:
        print("Run: fly-sniff-download-annotations")
    if not report["neuprint_token"]["present"]:
        print("Set NEUPRINT_TOKEN to enable live male-cns:v1.0 extraction.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check fly-sniff demo and MaleCNS prerequisites")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    report = diagnose()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)


if __name__ == "__main__":
    main()
