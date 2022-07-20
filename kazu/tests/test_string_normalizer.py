from kazu.utils.string_normalizer import StringNormalizer


def check_case(original, expected):
    result = StringNormalizer.normalize(original)
    assert result == expected


def test_normalizer():
    original = "MOP-2"
    expected = "MOP 2"
    check_case(original, expected)

    original = "y(+)L-type amino acid transporter 1"
    expected = "y (+) L TYPE AMINO ACID TRANSPORTER 1"
    check_case(original, expected)

    original = "mTOR"
    expected = "mTOR"
    check_case(original, expected)

    original = "egfr"
    expected = "EGFR"
    check_case(original, expected)

    original = "egfr(-)"
    expected = "EGFR (-)"
    check_case(original, expected)

    original = "C0D0C4J1X3"
    expected = "C0D0C4J1X 3"
    check_case(original, expected)

    original = "JAK-2"
    expected = "JAK 2"
    check_case(original, expected)

    original = "JAK 2"
    expected = "JAK 2"
    check_case(original, expected)

    original = "JAK2"
    expected = "JAK 2"
    check_case(original, expected)

    original = "MPNs"
    expected = "MPN s"
    check_case(original, expected)
