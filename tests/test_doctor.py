from fly_sniff.doctor import _interpreter_bin_dir, _resolve_console_script, diagnose


def test_doctor_reports_minimal_demo_readiness(tmp_path):
    report = diagnose(root=tmp_path, environ={}, ffmpeg_path="/usr/bin/ffmpeg")
    assert report["ready"]["gif_demo"]
    assert report["ready"]["mp4_showcase"]
    assert not report["ready"]["live_malecns"]
    assert not report["ready"]["offline_full_graph_trace"]


def test_console_script_resolution_prefers_active_interpreter_directory(tmp_path):
    script = tmp_path / "fly-sniff-trace"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o755)

    assert _resolve_console_script("fly-sniff-trace", tmp_path) == str(script)


def test_interpreter_bin_dir_does_not_follow_venv_python_symlink(tmp_path):
    base_bin = tmp_path / "base" / "bin"
    base_bin.mkdir(parents=True)
    real_python = base_bin / "python3.11"
    real_python.write_text("python")

    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    venv_python = venv_bin / "python"
    venv_python.symlink_to(real_python)

    assert _interpreter_bin_dir(venv_python) == venv_bin.absolute()
