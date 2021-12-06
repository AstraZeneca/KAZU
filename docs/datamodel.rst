Kazu data model
================================

The Kazu datamodel is based around the concepts of Documents and Steps. Steps are run over documents,
generally returning the original document with additional information added.


Documents are composed of a sequence of sections (for instance, title, body). A Section is a container
for text and metadata, (such as entities detected by an NER step).

.. code-block:: python

    from azner.data.data import SimpleDocument
    # a SimpleDocument is a subclass for Document for simple text strings
    step = SciSpacyAbbreviationExpansionStep([])
    doc = SimpleDocument("EGFR (Epidermal Growth Factor Receptor) is a gene")
    print(succeeded[0].get_text())

