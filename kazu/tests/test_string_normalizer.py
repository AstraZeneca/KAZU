import pytest

from kazu.utils.string_normalizer import StringNormalizer


@pytest.mark.parametrize(
    ("original", "expected", "entity_class"),
    (
        ("MOP-2", "MOP 2", "gene"),
        ("y(+)L-type amino acid transporter 1", "Y (+) L TYPE AMINO ACID TRANSPORTER 1", "gene"),
        ("mTOR", "MTOR", "gene"),
        ("egfr", "EGFR", "gene"),
        ("erbB2", "ERBB 2", "gene"),
        ("egfr(-)", "EGFR (-)", "gene"),
        ("C0D0C4J1X3", "C0D0C4J1X3", None),
        ("JAK-2", "JAK 2", "gene"),
        ("JAK2", "JAK 2", "gene"),
        ("MPNs", "MPN", "gene"),
        ("TESTIN gene", "TESTIN GENE", "gene"),
        (
            "Chromosome X",
            "CHROMOSOME X",
            None,
        ),  # Chromosome X and Chromosome 10 are different entities so don't normalize
    ),
)
def test_normalizer(original, expected, entity_class):
    result = StringNormalizer.normalize(original_string=original, entity_class=entity_class)
    assert result == expected
