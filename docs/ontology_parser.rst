Writing a custom OntologyParser
================================

*Ontologies are not designed for NLP* - Angus Roberts


Ontologies, or more broadly Knowledge bases are a core component of entity linking. In addition, they may hold a lot of
value as a vocabulary source for Dictionary based NER. However, they often need careful handling, as the (uncontextualised)
labels and synonyms associated with a given identifier can be noisy and/overloaded.

For instance, in the `MONDO ontology <https://www.ebi.ac.uk/ols/ontologies/mondo>`_, the abbreviation "OFD" is referenced as
has_exact_synonym for `osteofibrous dysplasia <http://purl.obolibrary.org/obo/MONDO_0011806>`_ and
`orofaciodigital syndrome <http://purl.obolibrary.org/obo/MONDO_0015375>`_ - i.e. two completely different diseases. Let's call
this scenario 1.

Similarly, the abbreviation "XLOA" is listed referenced as has_exact_symonym for `ocular albinism <http://purl.obolibrary.org/obo/MONDO_0017304>`_
and `X-linked recessive ocular albinism <http://purl.obolibrary.org/obo/MONDO_0021019>`_ - i.e. two very similar references. Let's call this scenario 2.

It gets worse... many ontologies make use of identifiers from other ontologies, as well as assigning their own. For instance, "D-TGA" refers to
"dextro-looped transposition of the great arteries" and actually has two identifiers associated:  `MONDO_0019443 <http://purl.obolibrary.org/obo/MONDO_0019443>`_
and `HP:0031348 <https://hpo.jax.org/app/browse/term/HP:0031348>`_ - i.e. the exact same thing, but with different ids. When we say "we will link to MONDO", what
do we mean "only MONDO ids" or "everything in the MONDO ontology"!?


Anyone familiar with a knowledgebase of any size will know such curation issues are not uncommon - When attempting to find candidates for
linking either "OFD", "XLOA" or "D-TGA" how should we reconcile these scenarios from an NLP perspective? For scenario 1, we can use some context of the underlying
text to model which IRI is more likely. However, for scenario 2, this is very difficult, as the ontology is telling us the abbreviation is valid for both
senses. We could arbitrarily choose one (or both). For scenario 3, it would seem like keeping both makes sense. Never the less, we need a system that can
handle all three scenarios.

Enter the Kazu :class:`kazu.modelling.ontology_preprocessing.base.OntologyParser`. The job of the OntologyParser is to transform an Ontology or Knowledgebase
into a set of :class:`kazu.data.data.SynonymTerm`. A :class:`kazu.data.data.SynonymTerm` is a container for a synonym, which understands what set of IDs the
synoynm may refer to, whether they refer to a single group of closely related concepts or multiple separate ones, and various other pieces of useful information
such as whether the term is symbolic (a.k.a. an abbreviation or some other identifier).

How does it work? When an ambiguous term is detected in the ontology, the parser must decide whether it should group the confused IDs into the same
:class:`kazu.data.data.EquivalentIdSet`, or different ones. The algorithm for doing this works as follows:

1) Use the :class:`kazu.utils.string_normalizer.StringNormalizer` to determine if the term is symbolic or not. If it's not symbolic (i.e. a noun phrase),
    merge the IDs into a single :class:`kazu.data.data.EquivalentIdSet`. The idea here is that noun phrase entities 'ought' to be distinct enough such that
    references to the same string across different identifiers refer to the same concept.symbolic

    example:
        'seborrheic eczema' -> IDs 'http://purl.obolibrary.org/obo/HP_0001051' and 'http://purl.obolibrary.org/obo/MONDO_0006608'
    result:
        EquivalentIdSetAggregationStrategy.MERGED_AS_NON_SYMBOLIC

2) if the term is symbolic, use the configured string scorer to determine if the default label associated with each instance of the term is above some predefined threshold.
    The idea here is that we can use embeddings to check if semantically, the confused symbol is referring to either a very similar concept, or something completely different
    in the knowledgebase. Typically, we use a distilled form of the `SapBert <https://github.com/cambridgeltl/sapbert>`_ model here, as it's very good at this.

    example:
        "OFD" -> either osteofibrous dysplasia (MONDO_0011806) or orofaciodigital syndrome (MONDO_0015375).
    result:
        sapbert similarity: 0.4532. Threshold: 0.70. Decision -> split into two instances of :class:`kazu.data.data.EquivalentIdSet`

    example:
        "XLOA" -> either X-linked recessive ocular albinism (MONDO_0021019 )or ocular albinism (MONDO_0017304)
    result:
        sapbert similarity: 0.7426. Threshold: 0.70. Decision -> merge into one instance of :class:`kazu.data.data.EquivalentIdSet`






