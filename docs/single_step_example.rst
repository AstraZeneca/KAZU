.. testcode::

    from kazu.data.data import Document
    from kazu.steps.string_preprocessing.scispacy_abbreviation_expansion import SciSpacyAbbreviationExpansionStep

    # creates a document with a single section
    doc = Document.create_simple_document("EGFR (Epidermal Growth Factor Receptor) is a gene")
    step = SciSpacyAbbreviationExpansionStep([])
    # a step may fail to process a document, so it returns two lists, successes and failures
    succeeded, failed = step([doc])
    print(succeeded[0].sections[0].get_text())

.. testoutput::
    :hide:

    Epidermal Growth Factor Receptor (Epidermal Growth Factor Receptor) is a gene
