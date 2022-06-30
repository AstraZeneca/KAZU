Kazu data model
================================

The Kazu datamodel is based around the concepts of Documents and Steps. Steps are run over documents,
generally returning the original document with additional information added.


Documents are composed of a sequence of sections (for instance, title, body). A Section is a container
for text and metadata, (such as entities detected by an NER step).

.. code-block:: python

    from kazu.data.data import Document
    from kazu.steps.string_preprocessing.scispacy_abbreviation_expansion import SciSpacyAbbreviationExpansionStep

    step = SciSpacyAbbreviationExpansionStep([])
    # creates a document with a single section
    doc = Document.create_simple_document("EGFR (Epidermal Growth Factor Receptor) is a gene")
    succeeded, failed = step([doc])
    print(succeeded[0].sections[0].get_text())
