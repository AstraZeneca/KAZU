Kazu data model
================================

The Kazu datamodel is based around the concepts of Documents and Steps. Steps are run over documents,
generally returning the original document with additional information added.


Documents are composed of a sequence of sections (for instance, title, body). A Section is a container
for text and metadata, (such as entities detected by an NER step).

.. include:: single_step_example.rst
