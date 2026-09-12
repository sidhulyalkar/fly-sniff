from fly_sniff.doctor import _resolve_console_script, diagnose


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
