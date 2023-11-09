Kazu Data Model
================================

The Kazu datamodel is based around the concepts of :class:`kazu.data.data.Document`\ s and :class:`kazu.steps.step.Step`\ s. Steps are run over documents,
generally returning the original document with additional information added.


Documents are composed of a sequence of :class:`kazu.data.data.Section`\ s (for instance: title, body). A :class:`~kazu.data.data.Section` is a container
for text and metadata (such as entities detected by an NER step).

.. include:: single_step_example.rst

For convenience, and to handle additional logging/failure events, Steps can be wrapped in a :class:`kazu.pipeline.pipeline.Pipeline`\ .

For further data model documentation, please see the API docs for :class:`kazu.data.data.Entity`, :class:`kazu.data.data.SynonymTerm` etc.

.. _data-serialization:

Data Serialization and deserialization
--------------------------------------

As :class:`~.Document`\\ s are the key container of data processed by (or to be
processed by) Kazu, we focus on the (de)serialization of this container, to and from
json format.

:meth:`.DocumentJsonUtils.doc_to_json_dict` is the key method here for serialization,
and :meth:`.Document.from_json` for deserialization.

:class:`~.Document` and other classes that can be stored on :class:`~.Document` have
a :meth:`~.Document.from_dict` method. Note that this method mutates the input dictionary
- if this is not desired for your usage, call :func:`copy.deepcopy` on your dictionary
before calling the relevant ``from_dict`` method.

These ``from_dict`` methods all expect dictionaries in the format produced by
:meth:`.DocumentJsonUtils.doc_to_json_dict` when serializing a document - for an object
like a :class:`~.Section` or :class:`~.Entity` this is the result of
:meth:`.DocumentJsonUtils.obj_to_dict_repr`.
