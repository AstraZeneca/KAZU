.. _curating_for_explosion:

Curating a knowledge base for NER and Linking
=============================================

Many entities in Biomedical NER do not require sophisticated NER or disambiguation techniques, as they are often
unambiguous, and have few genuine synonyms. For instance, terms such as "Breast Cancer" and "mitosis" can be taken at face value, and
simple string matching techniques can be employed (in our case, we use the `Spacy PhraseMatcher <https://spacy.io/api/phrasematcher>`_).

However, the string labels in ontologies tend to be noisy when taken 'wholesale', and need curation in order to ensure high precision matching.
For instance, the `Gene Ontology reference for envelope <http://amigo.geneontology.org/amigo/term/GO:0031975>`_ is highly ambiguous -
we wouldn't want this to be tagged every time we see the string 'envelope' appear in text. On the other hand
`cornified envelope assembly <http://amigo.geneontology.org/amigo/term/GO:1903575>`_ is highly specific, and whenever we see this string,
we can safely assume it refers to this GO id.

Given an ontology can contain 100 000s of labels, how do we curate these? It's too labour intensive to look at every one. Therefore, we
apply some pragmaticism in order to produce a set of precise labels we want to use for dictionary based NER and linking.

In Kazu, we take the following approach:

1. Generate synonym candidates from the raw ontology to build a putative list of terms we might want to use. Synonyms are generated
    via the :attr:`synonym_generator` to an instance of :class:`kazu.modelling.ontology_preprocessing.base.OntologyParser`, which can
    then be retrieved via the method :meth:`kazu.modelling.ontology_preprocessing.base.OntologyParser.generate_synonyms`\\. The
    generated strings can be retrieved from :attr:`kazu.data.data.SynonymTerm.terms`\\.
2. Build a pipeline from this list, execute this pipeline over a large corpora of target data, and explore the results to get a sense of
    which terms are 'noisy'. We recommend loading json serialised versions of :class:`kazu.data.data.Document` into a MongoDB instance for this,
    as this makes it very easy to navigate the data.
3. These hits can now be curated, by forming instance of :class:`kazu.data.data.Curation`. This class is a container for
    :class:`kazu.data.data.SynonymTermAction`, which offers a range of :class:`kazu.data.data.SynonymTermBehaviour`\\s,
    such as 'use this string NER and linking', or 'remove thhis string as a linking target. See the API documentation
    of :class:`kazu.data.data.Curation` for examples and details
4. You can also create an instance of :class:`kazu.data.data.GlobalParserActions` to modify the behaviour of a
    :class:`kazu.modelling.ontology_preprocessing.base.OntologyParser` in a 'global' sense, Again, see the API documentation for this

We provide some curations for common ontologies in the default model pack, within the 'curations' subdirectory, which are then
passed to the constructor of :class:`kazu.modelling.ontology_preprocessing.base.OntologyParser`. When
:meth:`kazu.modelling.ontology_preprocessing.base.OntologyParser.populate_databases` is called, these curation triggers are activated.
Note that this method also returns a 'cleaned' list of :class:`kazu.data.data.Curation`, which can then be used for dictionary based NER
(as is used for :class:`kazu.steps.joint_ner_and_linking.explosion.ExplosionStringMatchingStep`
