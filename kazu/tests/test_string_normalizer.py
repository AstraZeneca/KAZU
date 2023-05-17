import pytest

from kazu.utils.string_normalizer import StringNormalizer


@pytest.mark.parametrize(
    ("original", "expected"),
    (
        ("MOP-2", "MOP 2"),
        ("y(+)L-type amino acid transporter 1", "Y (+) L TYPE AMINO ACID TRANSPORTER 1"),
        ("mTOR", "MTOR"),
        ("egfr", "EGFR"),
        ("erbB2", "ERBB 2"),
        ("egfr(-)", "ERGR (-)"),
        ("C0D0C4J1X3", "C0D0C4J1X3"),
        ("JAK-2", "JAK 2"),
        ("JAK2", "JAK 2"),
        ("MPNs", "MPN"),
        ("TESTIN gene", "TESTIN GENE"),
    ),
)
def test_normalizer(original, expected):
    result = StringNormalizer.normalize(original_string=original, entity_class="gene")
    assert result == expected
