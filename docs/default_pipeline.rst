At a glance: How to use the default Kazu pipeline
-------------------------------------------------

For most use cases we've encountered, the default configuration should suffice. This will

1) Tag the following entity classes with a curated dictionary using the Spacy PhraseMatcher. This uses
   :py:class:`~kazu.steps.joint_ner_and_linking.explosion.ExplosionStringMatchingStep`

   a. gene
   b. disease
   c. drug
   d. cell_line
   e. cell_type
   f. gene ontology (split into go_bp, go_cc and go_mf)
   g. anatomy

2) Tag the following entity classes with the TinyBERN2 model (see the
   `EMNLP Kazu paper <https://aclanthology.org/2022.emnlp-industry.63>`_ for more details).
   This uses :py:class:`~kazu.steps.ner.hf_token_classification.TransformersModelForTokenClassificationNerStep`

   a. gene
   b. disease
   c. drug
   d. cell_line
   e. cell_type

3) Find candidates for linking the entities to knowledgebases according to the below yaml schema. This uses :py:class:`~kazu.steps.linking.dictionary.DictionaryEntityLinkingStep`

   .. code-block:: yaml

        drug:
          - CHEMBL
          - OPENTARGETS_MOLECULE
        disease:
          - MONDO
          - OPENTARGETS_DISEASE
        gene:
          - OPENTARGETS_TARGET
          - HGNC_GENE_FAMILY
        anatomy:
          - UBERON
        cell_line:
          - CELLOSAURUS
        cell_type:
          - CLO
          - CL
        go_bp:
          - BP_GENE_ONTOLOGY
        go_mf:
          - MF_GENE_ONTOLOGY
        go_cc:
          - CC_GENE_ONTOLOGY

4) Disambiguate the entity class of exactly overlappings entities within a document using :py:class:`~kazu.steps.linking.entity_class_disambiguation.EntityClassDisambiguationStep`

5) Decide which candidates are appropriate and extract mappings accordingly. This uses :py:class:`~kazu.steps.linking.post_processing.mapping_step.MappingStep`

6) Merge overlapping entities (where appropriate). This uses :py:class:`~kazu.steps.other.merge_overlapping_ents.MergeOverlappingEntsStep`

7) Detect abbreviations, and copy appropriate mapping information to the desired spans. This uses :py:class:`~kazu.steps.document_post_processing.abbreviation_finder.AbbreviationFinderStep`

8) Perform some customisable cleanup. This uses :py:class:`~kazu.steps.other.cleanup.CleanupStep`

All of these steps are customisable via Hydra configuration.

Note that other steps are available in Kazu which are not used in the default pipeline, such as:

- :py:class:`~kazu.steps.ner.seth.SethStep` for tagging mutations with the `SETH tagger <https://rockt.github.io/SETH/>`_.
- :py:class:`~kazu.steps.other.stanza.StanzaStep` for high accuracy sentence-segmentation (note that this does slow the pipeline down considerably, hence why it's not in by default).
- :py:class:`~kazu.steps.ner.spacy_ner.SpacyNerStep` for using a generic spacy pipeline (such as `scispacy <https://allenai.github.io/scispacy/>`_) for Named Entity Recognition.
