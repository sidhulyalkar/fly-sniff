from __future__ import annotations

from pathlib import Path
import subprocess

E002D_SCRIPTS = (
    Path("scripts/run_e002d_phase_crosswalk.sh"),
    Path("scripts/run_e002d_phase_probe.sh"),
    Path("scripts/render_showcase_v4.sh"),
    Path("scripts/run_e002d_and_render_v4.sh"),
)


def test_python_env_resolves_an_executable_interpreter() -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source scripts/python_env.sh; "$PYTHON_BIN" -c "import sys; print(sys.executable)"',
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip()


def test_e002d_scripts_use_resolved_python_not_bare_python() -> None:
    for path in E002D_SCRIPTS:
        text = path.read_text()
        assert "source scripts/python_env.sh" in text
        executable_lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        assert not any(
            line == "python" or line.startswith("python ") for line in executable_lines
        ), path
