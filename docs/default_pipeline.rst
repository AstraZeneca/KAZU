At a glance: How to use the default Kazu pipeline
-------------------------------------------------

For most use cases we've encountered, the default configuration should suffice. This will

1) Tag the following entity classes with a curated dictionary using the spaCy PhraseMatcher. This uses
   :py:class:`~kazu.steps.joint_ner_and_linking.memory_efficient_string_matching.MemoryEfficientStringMatchingStep`

   a. gene
   b. disease
   c. drug
   d. cell_line
   e. cell_type
   f. gene ontology (split into go_bp, go_cc and go_mf)
   g. anatomy

.. note::
   This step is limited to string matching only. A full `FlashText <https://github.com/vi3k6i5/flashtext>`_ implementation
   (i.e. based on tokens) is available via :py:class:`~kazu.steps.joint_ner_and_linking.explosion.ExplosionStringMatchingStep`,
   however this uses considerably more memory.

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

4) Apply rules to disambiguate certain entity classes and mentions within a document using :py:class:`~kazu.steps.linking.rules_based_disambiguation.RulesBasedEntityClassDisambiguationFilterStep`

5) Decide which candidates are appropriate and extract mappings accordingly. This uses :py:class:`~kazu.steps.linking.post_processing.mapping_step.MappingStep`

6) Merge overlapping entities (where appropriate). This uses :py:class:`~kazu.steps.other.merge_overlapping_ents.MergeOverlappingEntsStep`

7) Detect abbreviations, and copy appropriate mapping information to the desired spans. This uses :py:class:`~kazu.steps.document_post_processing.abbreviation_finder.AbbreviationFinderStep`

8) Perform some customisable cleanup. This uses :py:class:`~kazu.steps.other.cleanup.CleanupStep`

All of these steps are customisable via Hydra configuration.

Note that other steps are available in Kazu which are not used in the default pipeline, such as:

- :py:class:`~kazu.steps.ner.seth.SethStep` for tagging mutations with the `SETH tagger <https://rockt.github.io/SETH/>`_.
- :py:class:`~kazu.steps.ner.opsin.OpsinStep` for resolving IUPAC labels with the `OPSIN <https://opsin.ch.cam.ac.uk/>`_.
- :py:class:`~kazu.steps.other.stanza.StanzaStep` for high accuracy sentence-segmentation (note that this does slow the pipeline down considerably, hence why it's not in by default).
- :py:class:`~kazu.steps.ner.spacy_ner.SpacyNerStep` for using a generic spaCy pipeline (such as `scispacy <https://allenai.github.io/scispacy/>`_) for Named Entity Recognition.

Some of these require additional dependencies which are not included in the default installation of kazu. You can get all of these dependencies with:

.. code-block:: console

   $ pip install 'kazu[all-steps]'

Or you can install the specific required dependencies for just those steps out of the above that you are using - see their API docs for details.
