from collections.abc import Iterable

from kazu.data import Section, Entity
from kazu.utils.spacy_pipeline import SpacyPipelines, BASIC_PIPELINE_NAME, basic_spacy_pipeline
from spacy.tokens import Doc, Span, Token


class KazuToSpacyObjectMapper:
    """Maps entities and text from a :class:`.Section` to the spaCy data model using
    :func:`.basic_spacy_pipeline`\\.

    .. attention::
       Providing incomplete ``entity_classes`` for your usage (or leaving it blank)
       can lead to errors that might only occur infrequently when processing
       the results, and therefore may be difficult to track down.

       Therefore, users should be careful to set ``entity_classes`` to all the entity
       classes corresponding to attributes that they will access on the spaCy
       `Token <https://spacy.io/api/token>`_\\ s within the
       `Span <https://spacy.io/api/span>`_\\ s of the result of :meth:`.__call__`,
       whether directly or via spaCy `Matcher <https://spacy.io/api/matcher>`_ rules
       that check these custom attributes.


       The specific problem is that if you try to read a spaCy custom attribute that
       doesn't exist, you will get an error like::

          AttributeError: [E046] Can't retrieve unregistered extension attribute 'drug'.
          Did you forget to call the `set_extension` method?

       This class uses the provided ``entity_classes`` to call ``set_extension``. If
       the provided ``entity_classes`` is incomplete - say, missing ``"drug"`` - and
       you then try to access the ``drug`` attribute on a token in the result, you will
       get this error.
    """

    def __init__(
        self,
        entity_classes: Iterable[str] = set(),
        set_attributes_incrementally: bool = False,
    ):
        """

        :param entity_classes: known entity classes that the caller intends to access
            the spaCy extension attribute of with the result of :meth:`~.__call__`. See
            note above about the need to take care here.
        :param set_attributes_incrementally: whether to set a spaCy
            `custom extension attribute <https://spacy.io/usage/processing-pipelines#custom-components-attributes>`_
            for 'new' entity classes in :class:`.Section`\\ passed to
            :meth:`.__call__`. This will result in a more consistent result of
            ``__call__``, where every `Span <https://spacy.io/api/span>`_ in the
            dictionary will have an attribute for the relevant :class:`.Entity`'s entity
            class set to ``True`` for all the tokens in the span. However, it makes
            subtle bugs much more likely, so ``False`` is the default - see the note in
            the class-level docs if you are thinking about turning this on.
        """
        #: A set of entity classes known to this class. These will all have a spaCy
        #: `custom extension attribute <https://spacy.io/usage/processing-pipelines#custom-components-attributes>`_
        #: set. If ``set_attributes_incrementally`` is ``True``, as well as the
        #: ``entity_classes`` passed into the ``__init__``, this will include all
        #: entity classes encountered so far processing :class:`.Section`\ s passed in
        #: to :meth:`~.__call__`.
        self.entity_classes = set(entity_classes)
        self.set_attributes_incrementally = set_attributes_incrementally

        for entity_class in self.entity_classes:
            Token.set_extension(entity_class, default=False, force=True)
        self.spacy_pipelines = SpacyPipelines()
        self.spacy_pipelines.add_from_func(BASIC_PIPELINE_NAME, basic_spacy_pipeline)

    def __call__(self, section: Section) -> dict[Entity, Span]:
        """Convert a :class:`.Section` into a dictionary of :class:`.Entity` to
        spaCy `Span <https://spacy.io/api/span>`_\\ s."""

        spacy_doc: Doc = self.spacy_pipelines.process_single(section.text, BASIC_PIPELINE_NAME)
        ent_to_span = {}
        for entity in section.entities:
            entity_class = entity.entity_class
            span = spacy_doc.char_span(
                start_idx=entity.start,
                end_idx=entity.end,
                label=entity_class,
                alignment_mode="expand",
            )
            if span is not None:
                ent_to_span[entity] = span
                if entity_class not in self.entity_classes:
                    if self.set_attributes_incrementally:
                        Token.set_extension(entity_class, default=False, force=True)
                        self.entity_classes.add(entity_class)
                    else:
                        # we don't set the extension, or set the value of it
                        # on the individual tokens.
                        continue
                for token in span:
                    token._.set(entity_class, True)
        return ent_to_span
