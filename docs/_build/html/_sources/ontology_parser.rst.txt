The OntologyParser
================================

*Ontologies are not designed for NLP* - Angus Roberts


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

Enter the Kazu :class:`kazu.modelling.ontology_preprocessing.base.OntologyParser`. The job of the OntologyParser is to transform an Ontology or Knowledgebase
into a set of :class:`kazu.data.data.SynonymTerm`\ s. A :class:`.SynonymTerm` is a container for a synonym, which understands what set of IDs the
synonym may refer to and whether they refer to a single group of closely related concepts or multiple separate ones. This is handled by the attribute
:attr:`kazu.data.data.SynonymTerm.associated_id_sets`. A :class:`.SynonymTerm` holds various other pieces of useful information
such as whether the term is symbolic (i.e. an abbreviation or some other identifier).

How does it work? When an ambiguous term is detected in the ontology, the parser must decide whether it should group the confused IDs into the same
:class:`kazu.data.data.EquivalentIdSet`, or different ones. The algorithm for doing this works as follows:

1) Use the :class:`kazu.utils.string_normalizer.StringNormalizer` to determine if the term is symbolic or not. If it's not symbolic (i.e. a noun phrase),
   merge the IDs into a single :class:`kazu.data.data.EquivalentIdSet`. The idea here is that noun phrase entities 'ought' to be distinct enough such that
   references to the same string across different identifiers refer to the same concept.

    example:

        'seborrheic eczema' -> IDs 'http://purl.obolibrary.org/obo/HP_0001051' and 'http://purl.obolibrary.org/obo/MONDO_0006608'

    result:

        EquivalentIdSetAggregationStrategy.MERGED_AS_NON_SYMBOLIC

2) If the term is symbolic, use the configured string scorer to calculate the similarity of default labels associated with the different IDs, and using a predefined threshold,
   group these IDs into one or more sets of IDs. The idea here is that we can use embeddings to check if semantically, each ID associated with a confused symbol is referring
   to either a very similar concept to another ID associated with the symbol, or something completely different in the knowledgebase. Typically, we use a distilled form of the
   `SapBert <https://github.com/cambridgeltl/sapbert>`_ model here, as it's very good at this.

    example:

        "OFD" -> either `osteofibrous dysplasia <http://purl.obolibrary.org/obo/MONDO_0011806>`_ or `orofaciodigital syndrome <http://purl.obolibrary.org/obo/MONDO_0015375>`_.

    result:

        sapbert similarity: 0.4532. Threshold: 0.70. Decision -> split into two instances of :class:`kazu.data.data.EquivalentIdSet`

    example:

        "XLOA" -> either `X-linked recessive ocular albinism <http://purl.obolibrary.org/obo/MONDO_0021019>`_ or `ocular albinism <http://purl.obolibrary.org/obo/MONDO_0017304>`_

    result:

        sapbert similarity: 0.7426. Threshold: 0.70. Decision -> merge into one instance of :class:`kazu.data.data.EquivalentIdSet`

Naturally, this behaviour may not always be desired. You may want two instances of :class:`.SynonymTerm` for the term "XLOA" (despite the MONDO ontology
suggesting this abbreviation is appropriate for either ID), and allow another step to decide which candidate :class:`.SynonymTerm` is most appropriate.
In this case, you can override this behaviour with :meth:`kazu.modelling.ontology_preprocessing.base.OntologyParser.score_and_group_ids`\ .


Writing a Custom Parser
-------------------------

Say you want to make a parser for a new datasource, (perhaps for NER or as a new linking target). To do this, you need to write an :class:`.OntologyParser`.
Fortunately, this is generally quite easy to do. Let's take the example of the :class:`kazu.modelling.ontology_preprocessing.base.ChemblOntologyParser`.

There are two methods you need to override: :meth:`kazu.modelling.ontology_preprocessing.base.OntologyParser.parse_to_dataframe` and :meth:`kazu.modelling.ontology_preprocessing.base.OntologyParser.find_kb`. Let's look at the first of these:

.. code-block:: python

    import sqlite3

    import pandas as pd

    from kazu.modelling.ontology_preprocessing.base import (
        OntologyParser,
        DEFAULT_LABEL,
        IDX,
        SYN,
        MAPPING_TYPE,
    )


    def parse_to_dataframe(self) -> pd.DataFrame:
        """
        the objective of this method is to create a long, thin pandas dataframe of terms and associated metadata.
        We need at the very least, to extract an id and a default label. Normally, we'd also be looking to extract any
        synonyms and the type of mapping as well
        """

        # fortunately, Chembl comes as and sqlite DB, which lends itself very well to this tabular structure
        conn = sqlite3.connect(self.in_path)
        query = f"""
            SELECT chembl_id AS {IDX}, pref_name AS {DEFAULT_LABEL}, synonyms AS {SYN}, syn_type AS {MAPPING_TYPE}
            FROM molecule_dictionary AS md
                     JOIN molecule_synonyms ms ON md.molregno = ms.molregno
            UNION ALL
            SELECT chembl_id AS {IDX}, pref_name AS {DEFAULT_LABEL}, pref_name AS {SYN}, "pref_name" AS {MAPPING_TYPE}
            FROM molecule_dictionary
        """  # noqa
        df = pd.read_sql(query, conn)
        # eliminate anything without a pref_name, as will be too big otherwise
        df = df.dropna(subset=[DEFAULT_LABEL])

        df.drop_duplicates(inplace=True)

        return df

Secondly, we need to write the :meth:`kazu.modelling.ontology_preprocessing.base.OntologyParser.find_kb` method:

.. code-block:: python

    def find_kb(self, string: str) -> str:
        """
        in our case, this is very simple, as everything in the Chembl DB has a chembl based identifier
        Other ontologies may use composite identifiers, i.e. MONDO could contain native MONDO_xxxxx identifiers
        or HP_xxxxxxx identifiers. In this scenario, we'd need to parse the 'string' parameter of this method
        to extract the relevant KB identifier
        """
        return "CHEMBL"


Finally, we need to set the class `name` field, so the full class looks like:

.. code-block:: python

    class ChemblOntologyParser(OntologyParser):

        name = "CHEMBL"

        def find_kb(self, string: str) -> str:
            return "CHEMBL"

        def parse_to_dataframe(self) -> pd.DataFrame:
            conn = sqlite3.connect(self.in_path)
            query = f"""
                SELECT chembl_id AS {IDX}, pref_name AS {DEFAULT_LABEL}, synonyms AS {SYN}, syn_type AS {MAPPING_TYPE}
                FROM molecule_dictionary AS md
                         JOIN molecule_synonyms ms ON md.molregno = ms.molregno
                UNION ALL
                SELECT chembl_id AS {IDX}, pref_name AS {DEFAULT_LABEL}, pref_name AS {SYN}, "pref_name" AS {MAPPING_TYPE}
                FROM molecule_dictionary
            """
            df = pd.read_sql(query, conn)
            # eliminate anything without a pref_name, as will be too big otherwise
            df = df.dropna(subset=[DEFAULT_LABEL])

            df.drop_duplicates(inplace=True)

            return df

That's it! The datasource is now ready for integration into Kazu, and can be referenced as a mapping target or elsewhere.
