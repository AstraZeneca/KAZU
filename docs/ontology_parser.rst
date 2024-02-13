.. _ontology_parser:

The OntologyParser
================================

.. epigraph::
    *Ontologies are not designed for NLP*

    -- Angus Roberts

Ontologies, or more broadly Knowledge Bases are a core component of entity linking. In addition, they may hold a lot of
value as a vocabulary source for Dictionary based NER. However, they often need careful handling, as the (uncontextualised)
labels and synonyms associated with a given identifier can be noisy and/overloaded.

For instance, in the `MONDO ontology <https://www.ebi.ac.uk/ols/ontologies/mondo>`_, the abbreviation "OFD" is referenced as
has_exact_synonym for `osteofibrous dysplasia <http://purl.obolibrary.org/obo/MONDO_0011806>`_ and
`orofaciodigital syndrome <http://purl.obolibrary.org/obo/MONDO_0015375>`_ - i.e. two completely different diseases. Let's call
this scenario 1.

Similarly, the abbreviation "XLOA" is referenced as has_exact_synonym for `ocular albinism <http://purl.obolibrary.org/obo/MONDO_0017304>`_
and `X-linked recessive ocular albinism <http://purl.obolibrary.org/obo/MONDO_0021019>`_ - i.e. two very similar references. Let's call this scenario 2.

It gets worse... many ontologies make use of identifiers from other ontologies, as well as assigning their own. For instance, "D-TGA" refers to
"dextro-looped transposition of the great arteries" and actually has two identifiers associated:  `MONDO_0019443 <http://purl.obolibrary.org/obo/MONDO_0019443>`_
and `HP:0031348 <https://hpo.jax.org/app/browse/term/HP:0031348>`_ - i.e. the exact same thing, but with different ids. When we say "we will link to MONDO", do we mean "only MONDO ids" or "everything in the MONDO ontology"!?


Anyone familiar with a knowledgebase of any size will know such curation issues are not uncommon. When attempting to find candidates for
linking either "OFD", "XLOA" or "D-TGA" how should we reconcile these scenarios from an NLP perspective? For scenario 1, we can use some context of the underlying
text to model which ID in MONDO is more likely. However, for scenario 2, this is very difficult, as the ontology is telling us the abbreviation is valid for both
senses. We could arbitrarily choose one (or both). For scenario 3, it would seem like keeping both makes sense. Nevertheless, we need a system that can
handle all three scenarios.

Enter the Kazu :class:`.OntologyParser`. The job of the OntologyParser is to transform an Ontology or Knowledgebase
into a set of :class:`.SynonymTerm`\ s. A :class:`.SynonymTerm` is a container for a synonym, which understands what set of IDs the
synonym may refer to and whether they refer to a single group of closely related concepts or multiple separate ones. This is handled by the attribute
:attr:`.SynonymTerm.associated_id_sets`. A :class:`.SynonymTerm` holds various other pieces of useful information
such as whether the term is symbolic (i.e. an abbreviation or some other identifier).

How does it work? When an ambiguous term is detected in the ontology, the parser must decide whether it should group the confused IDs into the same
:class:`.EquivalentIdSet`, or different ones. The algorithm for doing this works as follows:

1) Use the :class:`.StringNormalizer` to determine if the term is symbolic or not. If it's not symbolic (i.e. a noun phrase),
   merge the IDs into a single :class:`.EquivalentIdSet`. The idea here is that noun phrase entities 'ought' to be distinct enough such that
   references to the same string across different identifiers refer to the same concept.

   Example:

   .. parsed-literal::

     "seborrheic eczema"
     IDs: "\ http://purl.obolibrary.org/obo/HP_0001051\ ",
          "\ http://purl.obolibrary.org/obo/MONDO_0006608\ "

   Result:

   .. parsed-literal::

     :attr:`.EquivalentIdAggregationStrategy.MERGED_AS_NON_SYMBOLIC`

2) If the term is symbolic, use the configured string scorer to calculate the similarity of default labels associated with the different IDs, and using a predefined threshold,
   group these IDs into one or more sets of IDs. The idea here is that we can use embeddings to check if semantically, each ID associated with a confused symbol is referring
   to either a very similar concept to another ID associated with the symbol, or something completely different in the knowledgebase. Typically, we use a distilled form of the
   `SapBert <https://github.com/cambridgeltl/sapbert>`_ model here, as it's very good at this.

   Example:

   .. parsed-literal::

     "OFD" either:
     `osteofibrous dysplasia <http://purl.obolibrary.org/obo/MONDO_0011806>`_
     `orofaciodigital syndrome <http://purl.obolibrary.org/obo/MONDO_0015375>`_

   Result:

   .. parsed-literal::

     sapbert similarity: 0.4532. Threshold: 0.70.
     Decision: split into two instances of :class:`.EquivalentIdSet`

   Example:

   .. parsed-literal::

     "XLOA" either:
     `X-linked recessive ocular albinism <http://purl.obolibrary.org/obo/MONDO_0021019>`_
     `ocular albinism <http://purl.obolibrary.org/obo/MONDO_0017304>`_

   Result:

   .. parsed-literal::

     sapbert similarity: 0.7426. Threshold: 0.70.
     Decision: merge into one instance of :class:`.EquivalentIdSet`

Naturally, this behaviour may not always be desired. You may want two instances of :class:`.SynonymTerm` for the term "XLOA" (despite the MONDO ontology
suggesting this abbreviation is appropriate for either ID), and allow another step to decide which candidate :class:`.SynonymTerm` is most appropriate.
In this case, you can override this behaviour with :meth:`.OntologyParser.score_and_group_ids`\ .

.. _writing-a-custom-parser:

Writing a Custom Parser
-------------------------

Say you want to make a parser for a new datasource, (perhaps for NER or as a new linking target). To do this, you need to write an :class:`.OntologyParser`.
Fortunately, this is generally quite easy to do. Let's take the example of the :class:`.ChemblOntologyParser`.

There are two methods you need to override: :meth:`.OntologyParser.parse_to_dataframe` and :meth:`.OntologyParser.find_kb`. Let's look at the first of these:

.. code-block:: python

    import sqlite3

    import pandas as pd

    from kazu.ontology_preprocessing.base import (
        OntologyParser,
        DEFAULT_LABEL,
        IDX,
        SYN,
        MAPPING_TYPE,
    )


    def parse_to_dataframe(self) -> pd.DataFrame:
        """The objective of this method is to create a long, thin pandas dataframe of terms and
        associated metadata.

        We need at the very least, to extract an id and a default label. Normally, we'd also be
        looking to extract any synonyms and the type of mapping as well.
        """

        # fortunately, Chembl comes as an sqlite DB,
        # which lends itself very well to this tabular structure
        conn = sqlite3.connect(self.in_path)
        query = f"""\
            SELECT chembl_id AS {IDX}, pref_name AS {DEFAULT_LABEL}, synonyms AS {SYN},
                syn_type AS {MAPPING_TYPE}
            FROM molecule_dictionary AS md
                     JOIN molecule_synonyms ms ON md.molregno = ms.molregno
            UNION ALL
            SELECT chembl_id AS {IDX}, pref_name AS {DEFAULT_LABEL}, pref_name AS {SYN},
                'pref_name' AS {MAPPING_TYPE}
            FROM molecule_dictionary
        """
        df = pd.read_sql(query, conn)
        # eliminate anything without a pref_name, as will be too big otherwise
        df = df.dropna(subset=[DEFAULT_LABEL])

        df.drop_duplicates(inplace=True)

        return df

Secondly, we need to write the :meth:`.OntologyParser.find_kb` method:

.. code-block:: python

    def find_kb(self, string: str) -> str:
        """In our case, this is simple, as everything in the Chembl DB has a chembl identifier.

        Other ontologies may use composite identifiers, e.g. MONDO contains native MONDO_xxxxx
        identifiers as well as HP_xxxxxxx identifiers. In this scenario, we'd need to parse the
        'string' parameter of this method to extract the relevant KB identifier.
        """
        return "CHEMBL"


The full class looks like:

.. code-block:: python

    class ChemblOntologyParser(OntologyParser):
        def find_kb(self, string: str) -> str:
            return "CHEMBL"

        def parse_to_dataframe(self) -> pd.DataFrame:
            conn = sqlite3.connect(self.in_path)
            query = f"""\
                SELECT chembl_id AS {IDX}, pref_name AS {DEFAULT_LABEL}, synonyms AS {SYN},
                    syn_type AS {MAPPING_TYPE}
                FROM molecule_dictionary AS md
                         JOIN molecule_synonyms ms ON md.molregno = ms.molregno
                UNION ALL
                SELECT chembl_id AS {IDX}, pref_name AS {DEFAULT_LABEL}, pref_name AS {SYN},
                    'pref_name' AS {MAPPING_TYPE}
                FROM molecule_dictionary
            """
            df = pd.read_sql(query, conn)
            # eliminate anything without a pref_name, as will be too big otherwise
            df = df.dropna(subset=[DEFAULT_LABEL])

            df.drop_duplicates(inplace=True)

            return df

Finally, when we want to use our new parser, we need to give it information about what entity class it is associated with:

.. code-block:: python

    # We need a string scorer to resolve similar terms.
    # Here, we use a trivial example for brevity.
    string_scorer = lambda string_1, string_2: 0.75
    parser = ChemblOntologyParser(
        in_path="path to chembl DB goes here",
        # if used in entity linking, entities with class 'drug'
        # will be associated with this parser
        entity_class="drug",
        name="CHEMBL",  # a globally unique name for the parser
        string_scorer=string_scorer,
    )

That's it! The datasource is now ready for integration into Kazu, and can be referenced as a linking target or elsewhere.

Using an Ontology for dictionary based matching
-------------------------------------------------

The data sources that Kazu users tend to concern themselves are often a rich source of nouns that can be accurately used for
dictionary based string matching. Naively, we might think it is sufficient to simply take all of the entity labels from an
ontology, and perform case insensitive string matching with them. However, unless we have direct control over the ontology,
this is rarely the case.

Instead, it's preferable to curate the ontology, specifying:

1) Strings we want to use from the ontology, and strings we want to ignore.
2) Strings that we want to use for dictionary matching and entity linking, or just entity linking.
3) Whether the case of the string is relevant.
4) How confident one is that a given string match is likely to be a 'true positive' entity hit

In addition, there are the following considerations:

5) Many strings have multiple equally relevant forms/synonyms that can be automatically generated. How can we
    ensure we are using those for NER/linking as well?
6) If the ontology is large, it's probably not practical to review every string - there could be 10 000s.
    Therefore, can we employ heuristics to automatically curate some/all of the strings for us?
7) Usually, ontologies are not static. They undergo revisions, in which new strings are added, obsolete ones removed
    and existing ones change. Even with autocuration techniques, some manual review will probably be necessary. How can
    we preserve the work of our previous round of curation, when a new version of an ontology is released?
8) Curations can clash! The behaviour of one may interfere with another, similar curation. How can we ensure behaviour
    is consistent across our set of curations?

.. note::
    Prior to Kazu 2.0, the internal curation system of Kazu was cumbersome to use/explain. We recommend upgrading to
    Kazu 2.0 or later as soon as possible.

Points 1-4 above are handled by the :class:`.CuratedTerm` concept and :class:`.MentionForm` concept. Point 5 is handled by
the :class:`.CombinatorialSynonymGenerator` class. Point 6 is handled by the :class:`.Autocurator` class. Point 7 is
handled by :meth:`.OntologyParser.generate_clean_default_curations` (and controlled by the `run_upgrade_report` flag).
Point 8 is handled by the :class:`.CuratedTermConflictAnalyser` class (and controlled by the `run_curation_report` flag).

The flow of an ontology parser to handling the underlying strings is as follows:

1) On first initialisation, the set of :class:`.SynonymTerm`s an ontology produces is converted into a set of
    :class:`.CuratedTerm`.
2) If configured, the :class:`.CombinatorialSynonymGenerator` is executed to generate additional forms
    for each :class:`.CuratedTerm`.
3) If configured, the :class:`.Autocurator` is executed to set the default behaviour for each :class:`.CuratedTerm`
4) The final set of the automatically generated :class:`.CuratedTerm`s is serialised in the model pack. This
    is required when upgrading to a new version of the ontology, and can also be used as the basis for human curations
    (supplied via a seperate file to the `curations_path` argument to :class:`.OntologyParser`
5) The automatically generated set of :class:`.CuratedTerm` is guaranteed to be consistent. However, it can be difficult
    to determine whether any additional human curations will cause a conflict. Therefore, the
    :class:`.CuratedTermConflictAnalyser` will run each time the :meth:`.OntologyParser.populate_databases` method is
    called (once per python process, or as long as `force=True`). This will throw an exception in the case of conflicts,
    describing the human curations that need to be adjusted.
6) Finally, when upgrading an ontology, the serialised set of automatically produced :class:`.CuratedTerm` from step 4
    is used to compare the new and old ontologies, migrating terms where possible and describing obsolete curations (both
    automatically generated and human). The results are summarised in a report inside the Kazu model pack, alongside the
    original ontology input data.

To explore the other capabilities of the :class:`.OntologyParser`, such as synonym generation and ID filtering, please
refer to the API documentation.
