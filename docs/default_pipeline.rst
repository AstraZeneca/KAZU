At a glance: How to use Kazu
------------------------------

For most use cases we've encountered, the default configuration should suffice. This will

1) tag the following entity classes with the TinyBERN2 model (see the EMNLP Kazu paper for more details - Link TBA ). This uses
    :py:class:`kazu.steps.ner.hf_token_classification.TransformersModelForTokenClassificationNerStep`

    a. gene
    b. disease
    c. drug
    d. cell_line
    e. cell_type

2) tag the following entity classes with a curated dictionary using the Spacy PhraseMatcher. This uses
    :py:class:`kazu.steps.joint_ner_and_linking.explosion.ExplosionStringMatchingStep`

    a. gene
    b. disease
    c. drug
    d. cell_line
    e. cell_type
    f. gene ontology (split into go_bp,go_cc and go_mf)
    g. anatomy

3) tag mutations with the `SETH tagger <https://rockt.github.io/SETH/>`_. This uses :py:class:`kazu.steps.ner.seth.SethStep`

4) Find candidates for linking the entities to knowledgebases according to the below yaml schema. This uses :py:class:`kazu.steps.linking.dictionary.DictionaryEntityLinkingStep`

.. code-block:: yaml

    drug:
      - CHEMBL
      - OPENTARGETS_MOLECULE
    disease:
      - MONDO
      - OPENTARGETS_DISEASE
    gene:
      - OPENTARGETS_TARGET
    anatomy:
      - UBERON
    cell_line:
      - CELLOSAURUS
    cell_type:
      - CLO
    go_bp:
      - BP_GENE_ONTOLOGY
    go_mf:
      - MF_GENE_ONTOLOGY
    go_cc:
      - CC_GENE_ONTOLOGY

5) decide which candidates are appropriate and extract mappings accordingly. This uses :py:class:`kazu.steps.linking.mapping_step.MappingStep`

6) merge overlapping entities (where appropriate). This uses :py:class:`kazu.steps.other.merge_overlapping_ents.MergeOverlappingEntsStep`

7) Perform some customisable cleanup. This uses :py:class:`kazu.steps.other.cleanup.CleanupStep`

8) Detect abbreviations, and copy appropriate mapping information to the desired spans. This uses :py:class:`kazu.steps.document_post_processing.abbreviation_finder.AbbreviationFinderStep`

All of these steps are customisable via Hydra configuration
