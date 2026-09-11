import pytest

from fly_sniff.bulk_data import require_large_download_confirmation


def test_large_download_requires_explicit_confirmation():
    with pytest.raises(SystemExit, match="roughly 1.1 GB"):
        require_large_download_confirmation(False)


def test_large_download_confirmation_passes():
    require_large_download_confirmation(True)
