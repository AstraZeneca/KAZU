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

1. Generate synonym candidates from the raw ontology to build a putative list of terms we might want to use. If the term is symbolic,
   we assume it's case sensitive. Otherwise assume case insensitive.
2. Build a pipeline from this list, execute this pipeline over a large corpora of target data, and explore the results to get a sense of
   which terms are 'noisy'
3. Curate the top x hits by frequency, to determine whether a given term is precise enough in it's own right to be valid for dictionary based NER.
   We assume here that if a term doesn't hit frequently enough to be considered in step 2, it's probably safe to include. Depending on your target
   data, this may be invalid -  so in practice, the curation approach is iterative.

TODO: add a worked example
