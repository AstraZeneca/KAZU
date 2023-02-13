Kazu Data Model
================================

The Kazu datamodel is based around the concepts of :class:`kazu.data.data.Document`\ s and :class:`kazu.steps.step.Step`\ s. Steps are run over documents,
generally returning the original document with additional information added.


Documents are composed of a sequence of :class:`kazu.data.data.Section`\ s (for instance: title, body). A :class:`kazu.data.data.Section` is a container
for text and metadata (such as entities detected by an NER step).

.. include:: single_step_example.rst

For convenience, and to handle additional logging/failure events, Steps can be wrapped in a :class:`kazu.pipeline.pipeline.Pipeline`\ .

For further data model documentation, please see the API docs for :class:`kazu.data.data.Entity`, :class:`kazu.data.data.SynonymTerm` etc.
