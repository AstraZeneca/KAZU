.. testcode::

    from kazu.data import Document, Entity
    from kazu.steps.document_post_processing.abbreviation_finder import (
        AbbreviationFinderStep,
    )

    # creates a document with a single section
    doc = Document.create_simple_document(
        "Epidermal Growth Factor Receptor (EGFR) is a gene."
    )
    # create an Entity for the span "Epidermal Growth Factor Receptor"
    entity = Entity.load_contiguous_entity(
        # start and end are the character indices for the entity
        start=0,
        end=len("Epidermal Growth Factor Receptor"),
        namespace="example",
        entity_class="gene",
        match="Epidermal Growth Factor Receptor",
    )

    # add it to the documents first (and only) section
    doc.sections[0].entities.append(entity)

    # create an instance of the AbbreviationFinderStep
    step = AbbreviationFinderStep()
    # a step may fail to process a document, so it returns two lists:
    # all the docs, and just the failures
    processed, failed = step([doc])
    # check that a new entity has been created, attached to the EGFR span
    egfr_entity = next(filter(lambda x: x.match == "EGFR", doc.get_entities()))
    assert egfr_entity.entity_class == "gene"
    print(egfr_entity.match)

.. testoutput::
    :hide:

    EGFR
