from kazu.utils.string_normalizer import StringNormalizer


def check_case(original, expected, entity_class):
    result = StringNormalizer.normalize(original_string=original, entity_class=entity_class)
    assert result == expected


def test_normalizer():
    original = "MOP-2"
    expected = "MOP 2"
    check_case(original, expected, "gene")

    original = "y(+)L-type amino acid transporter 1"
    expected = "Y (+) L TYPE AMINO ACID TRANSPORTER 1"
    check_case(original, expected, "gene")

    original = "mTOR"
    expected = "MTOR"
    check_case(original, expected, "gene")

    original = "egfr"
    expected = "EGFR"
    check_case(original, expected, "gene")

    original = "erbB2"
    expected = "ERBB 2"
    check_case(original, expected, "gene")

    original = "egfr(-)"
    expected = "EGFR (-)"
    check_case(original, expected, "gene")

    original = "C0D0C4J1X3"
    expected = "C0D0C4J1X3"
    check_case(original, expected, None)

    original = "JAK-2"
    expected = "JAK 2"
    check_case(original, expected, "gene")

    original = "JAK 2"
    expected = "JAK 2"
    check_case(original, expected, "gene")

    original = "JAK2"
    expected = "JAK 2"
    check_case(original, expected, "gene")

    original = "MPNs"
    expected = "MPN"
    check_case(original, expected, "gene")

    original = "TESTIN gene"
    expected = "TESTIN GENE"
    check_case(original, expected, "gene")
