Kazu data model
================================

The Kazu datamodel is based around the concepts of :class:`kazu.data.data.Document`\ s and Steps (instances of :class:`kazu.steps.base.step.BaseStep`\ ). Steps are run over documents,
generally returning the original document with additional information added.


Documents are composed of a sequence of :class:`kazu.data.data.Section` (for instance, title, body). A :class:`kazu.data.data.Section` is a container
for text and metadata, (such as entities detected by an NER step).

.. include:: single_step_example.rst

For convenience, and to handle additional logging/failure events, Steps can be wrapped in a :class:`kazu.pipeline.pipeline.Pipeline`\ ).

Entity
-------

A :class:`kazu.data.data.Entity` is a container for information about a single entity detected within a :class:`kazu.data.data.Section`

Within an :class:`kazu.data.data.Entity`, the most important fields are :attr:`kazu.data.data.Entity.match` (the actual string detected),
:attr:`kazu.data.data.Entity.syn_term_to_synonym_terms`, a dict of :class:`kazu.data.data.SynonymTermWithMetrics` (candidates for knowledgebase hits)
and :attr:`kazu.data.data.Entity.mappings`, the final product of linked references to the underlying entity
