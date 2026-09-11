from fly_sniff.doctor import diagnose


def test_doctor_reports_minimal_demo_readiness(tmp_path):
    report = diagnose(root=tmp_path, environ={}, ffmpeg_path="/usr/bin/ffmpeg")
    assert report["ready"]["gif_demo"]
    assert report["ready"]["mp4_showcase"]
    assert not report["ready"]["live_malecns"]
    assert not report["ready"]["offline_full_graph_trace"]
