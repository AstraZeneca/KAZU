from kazu.data.data import TokenizedWord, SimpleDocument, CharSpan, Entity


def test_tokenized_word():
    # fillers used to complete params, but not needed in tests
    word_label_strings = ["B-gene", "I-gene", "I-gene"]
    expected_bio_labels = ["B", "I", "I"]
    expected_class_labels = ["gene", "gene", "gene"]
    word_labels = [0 for _ in range(len(word_label_strings))]
    word_confidences = [0.5 for _ in range(len(word_label_strings))]
    word_offsets = [
        (
            0,
            1,
        )
        for _ in range(len(word_label_strings))
    ]
    word = TokenizedWord(
        word_labels=word_labels,
        word_labels_strings=word_label_strings,
        word_offsets=word_offsets,
        word_confidences=word_confidences,
    )
    assert len(word.bio_labels) == 0
    assert len(word.class_labels) == 0
    word.parse_labels_to_bio_and_class()
    assert len(word.bio_labels) == len(word_label_strings)
    assert len(word.class_labels) == len(word_label_strings)
    for expected, observed in zip(expected_bio_labels, word.bio_labels):
        assert expected == observed
    for expected, observed in zip(expected_class_labels, word.class_labels):
        assert expected == observed


def test_serialisation():
    x = SimpleDocument("Hello")
    x.sections[0].offset_map = {CharSpan(start=1, end=2): CharSpan(start=1, end=2)}
    x.sections[0].entities = [
        Entity(
            namespace="test",
            match="metastatic liver cancer",
            entity_class="test",
            spans=frozenset([CharSpan(start=16, end=39)]),
        )
    ]
    x.json()


def test_overlap_logic():
    # e.g. "the patient has metastatic liver cancers"
    e1 = Entity(
        namespace="test",
        match="metastatic liver cancer",
        entity_class="test",
        spans=frozenset([CharSpan(start=16, end=39)]),
    )
    e2 = Entity(
        namespace="test",
        match="liver cancers",
        entity_class="test",
        spans=frozenset([CharSpan(start=27, end=40)]),
    )

    assert e1.is_partially_overlapped(e2)

    # e.g. 'liver and lung cancer'
    e1 = Entity(
        namespace="test",
        match="liver cancer",
        entity_class="test",
        spans=frozenset([CharSpan(start=0, end=4), CharSpan(start=15, end=21)]),
    )
    e2 = Entity(
        namespace="test",
        match="lung cancer",
        entity_class="test",
        spans=frozenset([CharSpan(start=9, end=21)]),
    )
    assert not e1.is_partially_overlapped(e2)
