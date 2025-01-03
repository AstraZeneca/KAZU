import unicodedata
from sys import maxunicode

from omegaconf import DictConfig
import pytest
from kazu.data import (
    OntologyStringResource,
    MentionConfidence,
    OntologyStringBehaviour,
    Synonym,
)
from kazu.language.language_phenomena import GREEK_SUBS
from kazu.ontology_preprocessing.synonym_generation import (
    SeparatorExpansion,
    SynonymGenerator,
    StopWordRemover,
    StringReplacement,
    GreekSymbolSubstitution,
    CombinatorialSynonymGenerator,
    VerbPhraseVariantGenerator,
    TokenListReplacementGenerator,
)
from kazu.tests.utils import requires_model_pack


def check_generator_result(
    input_str: str,
    expected_syns: set[str],
    generator: SynonymGenerator,
) -> None:
    new_syns = generator(input_str)
    assert new_syns == expected_syns


@pytest.mark.parametrize(
    argnames=("input_str", "expected_syns"),
    argvalues=(
        (
            "ABAC (ABAC1/ABAC2)",
            {"ABAC", "ABAC1", "ABAC2", "ABAC1/ABAC2", "ABAC ABAC1/ABAC2"},
        ),
        (
            "cyclin-dependent kinase inhibitor 1B (p27, Kip1)",
            {
                "cyclin-dependent kinase inhibitor 1B",
                "p27",
                "Kip1",
                "p27, Kip1",
                "cyclin-dependent kinase inhibitor 1B p27, Kip1",
            },
        ),
        (
            "gonadotropin-releasing hormone (type 2) receptor 2",
            {"gonadotropin-releasing hormone receptor 2"},
        ),
        (
            "oxidase (cytochrome c) assembly 1-like",
            {"oxidase assembly 1-like"},
        ),
    ),
)
def test_SeparatorExpansion(input_str: str, expected_syns: set[str]) -> None:
    check_generator_result(input_str, expected_syns, SeparatorExpansion())


def test_StopWordRemover() -> None:
    generator = StopWordRemover()
    check_generator_result(
        input_str="The cat sat in the mat",
        expected_syns={"cat sat mat"},
        generator=generator,
    )


def test_StringReplacement() -> None:
    generator = StringReplacement(replacement_dict={"cat": ["dog", "chicken"]})
    check_generator_result(
        input_str="The cat sat on the mat",
        expected_syns={
            "The dog sat on the mat",
            "The chicken sat on the mat",
        },
        generator=generator,
    )


@pytest.fixture(scope="session")
def greek_symbol_generator() -> StringReplacement:
    return StringReplacement(include_greek=True)


@pytest.mark.parametrize(
    argnames=("input_str", "expected_syns"),
    argvalues=(
        (
            "alpha-thalassaemia",
            {"α-thalassaemia", "Α-thalassaemia"},
        ),
        (
            "α-thalassaemia",
            {"alpha-thalassaemia", "a-thalassaemia", "Α-thalassaemia"},
        ),
        (
            "A-thalassaemia",
            set(),
        ),
        pytest.param(
            "beta test",
            {
                "β test",
                "ϐ test",
                "Β test",
            },
            marks=pytest.mark.xfail(reason="'beta' also gets substring 'eta' replaced"),
        ),
        pytest.param(
            "alpha beta test",
            {
                "alpha β test",
                "alpha ϐ test",
                "alpha Β test",
                "α beta test",
                "α β test",
                "α ϐ test",
                "α Β test",
                "Α beta test",
                "Α β test",
                "Α ϐ test",
                "Α Β test",
            },
            marks=pytest.mark.xfail(reason="above problem with beta/eta"),
        ),
    ),
)
def test_GreekSymbolSubstitution(
    input_str: str, expected_syns: set[str], greek_symbol_generator: StringReplacement
) -> None:
    check_generator_result(input_str, expected_syns, greek_symbol_generator)


def test_TokenListReplacementGenerator() -> None:
    generator = TokenListReplacementGenerator(
        token_lists_to_consider=[["typical", "ordinary"], ["abnormal", "incorrect"]],
    )
    input_str = "ALT was typical"
    expected_syns = {"ALT was ordinary", input_str}
    check_generator_result(input_str, expected_syns, generator)
    input_str = "ALT was abnormal"
    expected_syns = {"ALT was incorrect", input_str}
    check_generator_result(input_str, expected_syns, generator)


@requires_model_pack
def test_VerbPhraseVariantGenerator(kazu_test_config: DictConfig) -> None:
    generator = VerbPhraseVariantGenerator(
        tense_templates=["{NOUN} {TARGET}", "{TARGET} in {NOUN}"],
        spacy_model_path=kazu_test_config.SciSpacyPipeline.path,
        lemmas_to_consider={
            "increase": ["increasing", "increased"],
            "decrease": ["decreasing", "decreased"],
        },
    )
    input_str = "ALT increased"
    expected_syns = {
        # the input string isn't 'inherently' included
        # in the output, but happens to be produced
        # in this case
        input_str,
        "ALT increasing",
        "ALT increase",
        "increased in ALT",
        "increasing in ALT",
        "increase in ALT",
    }
    check_generator_result(input_str, expected_syns, generator)
    input_str = "decreasing ALT"
    expected_syns = {
        "ALT decreasing",
        "ALT decrease",
        "ALT decreased",
        "decreased in ALT",
        "decreasing in ALT",
        "decrease in ALT",
    }
    check_generator_result(input_str, expected_syns, generator)


def test_greek_substitution_is_stripped() -> None:
    for k, val_set in GreekSymbolSubstitution.ALL_SUBS.items():
        assert k.strip() == k
        assert all(val.strip() == val for val in val_set)


@pytest.mark.xfail(
    reason="awkward casing behaviour where there are e.g. multiple uppercase theta's in unicode."
)
def test_greek_substitution_dict_casing() -> None:
    for k, val_set in GreekSymbolSubstitution.ALL_SUBS.items():
        for v in val_set:
            # if we substitute to a greek letter, we should sub
            # to both cases
            if v in GREEK_SUBS:
                if v.islower():
                    flipped_case_v = v.upper()
                elif v.isupper():
                    flipped_case_v = v.lower()
                assert flipped_case_v in GREEK_SUBS
                assert flipped_case_v == k or flipped_case_v in val_set


@pytest.mark.xfail(reason="haven't covered (or marked as exceptions) all unicode greek chars yet")
def test_greek_substitution_dict_uncode_variants() -> None:
    for i in range(maxunicode):
        char = chr(i)
        try:
            char_name_lower = unicodedata.name(char).lower()
            for greek_letter in GREEK_SUBS.values():
                assert greek_letter not in char_name_lower
        except ValueError:
            # character doesn't exist or has no name
            pass


@pytest.mark.parametrize(
    argnames=("resource", "expected_syns", "generators"),
    argvalues=(
        (
            OntologyStringResource(
                original_synonyms=frozenset(
                    [
                        Synonym(
                            text="alpha-thalassaemia",
                            mention_confidence=MentionConfidence.PROBABLE,
                            case_sensitive=False,
                        )
                    ]
                ),
                behaviour=OntologyStringBehaviour.ADD_FOR_NER_AND_LINKING,
            ),
            {
                "alpha thalassaemia",
                "alpha_thalassaemia",
                "α-thalassaemia",
                "α thalassaemia",
                "α_thalassaemia",
                "Α-thalassaemia",
                "Α thalassaemia",
                "Α_thalassaemia",
            },
            [
                StringReplacement(include_greek=True),
                StringReplacement(replacement_dict={"-": [" ", "_"]}, include_greek=False),
            ],
        ),
        (
            OntologyStringResource(
                original_synonyms=frozenset(
                    [
                        Synonym(
                            text="estimate",
                            mention_confidence=MentionConfidence.PROBABLE,
                            case_sensitive=False,
                        ),
                        Synonym(
                            text="estimate of",
                            mention_confidence=MentionConfidence.PROBABLE,
                            case_sensitive=False,
                        ),
                    ]
                ),
                behaviour=OntologyStringBehaviour.ADD_FOR_NER_AND_LINKING,
            ),
            set(),
            [
                StopWordRemover(),  # normally stopword remover would remove 'of' but since "estimate" is an original synonym
                # CombinatorialSynonymGenerator will skip the alternative synonym generation
            ],
        ),
        (
            OntologyStringResource(
                original_synonyms=frozenset(
                    [
                        Synonym(
                            text="estimate of",
                            mention_confidence=MentionConfidence.PROBABLE,
                            case_sensitive=False,
                        ),
                    ]
                ),
                behaviour=OntologyStringBehaviour.ADD_FOR_NER_AND_LINKING,
                alternative_synonyms=frozenset(
                    [
                        Synonym(
                            text="estimate",
                            mention_confidence=MentionConfidence.PROBABLE,
                            case_sensitive=False,
                        )
                    ]
                ),
            ),
            {"estimate"},
            [
                StopWordRemover(),
            ],
        ),
    ),
)
def test_CombinatorialSynonymGenerator(
    resource: OntologyStringResource, expected_syns: set[str], generators: list[SynonymGenerator]
) -> None:
    generator = CombinatorialSynonymGenerator(generators)
    generated_resources = generator({resource})
    new_syns = {s.text for resource in generated_resources for s in resource.alternative_synonyms}
    assert new_syns == expected_syns
