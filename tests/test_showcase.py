from fly_sniff.party_social import PROXY_CLAIM_LABEL
from fly_sniff.showcase import SHOWCASE_ASPECT, SHOWCASE_HEIGHT, SHOWCASE_WIDTH


def test_showcase_is_primary_four_by_five_social_format():
    assert (SHOWCASE_WIDTH, SHOWCASE_HEIGHT) == (1080, 1350)
    assert SHOWCASE_ASPECT == 4 / 5


def test_showcase_remains_development_only():
    assert "NOT A MALECNS RESULT" in PROXY_CLAIM_LABEL
