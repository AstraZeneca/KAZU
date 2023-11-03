Introduction
============

Why Kazu?
------------

Kazu is a lightweight biomedical NER and Linking (also known as 'grounding' or 'normalisation') pipelining framework used at AstraZeneca. Some of the components have been
developed via our collaboration with the DMIS Lab at Korea University, whilst others are reworkings and integrations with a plethora of concepts from the BioNLP and wider
NLP community over the last 20 years.

But why make another NLP framework, when there's already so many great NLP frameworks out there? Our focus with Kazu is specifically biomedical literature, and our
experience tells us that none of the existing frameworks have the native support for the language phenomena that makes the BioNLP domain particularly challenging.

Specifically, when we set out to make Kazu, we wanted to ensure we had a data model that supports the following concepts:

For NER:

1) Non-contiguous entities (see `Extending TextAE for annotation of non-contiguous entities <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7362949/>`_).
2) Nested entities (see `Recognizing Nested Named Entities in GENIA corpus <https://aclanthology.org/W06-3318.pdf>`_).

For linking:

1) There are numerous ontologies that overlap, either conceptually or in a composite fashion. Rather than depending on a composite system like UMLS,
   we prefer to link directly to the source ontology URIs. This enables Kazu to always be up to date with a given ontology, whilst avoiding issues
   associated with the development of a composite ontology.

2) However, entity Linking is a challenging problem that starts with dealing with the inconsistencies with the source knowledgebase.
   Therefore, we wanted a system that could fully (or at least partially) automate the preprocessing/cleaning of knowledgebases and
   ontologies, in preparation for them to become a linking target.

Finally, speed is important. The intention with Kazu is that it will process and reprocess millions of documents. Therefore, the system should be able to
process documents efficiently and scale easily (i.e. without requiring expensive GPU acceleration).

Regarding the actual models and algorithms in Kazu, the framework includes several well regarded and state of the art approaches by
default. However, we recognise that NLP is a fast moving field. Therefore, our principal interest is extensibility, maintainable code and the isolation of
concepts, such that new developments can be brought into Kazu with little work.

Summary
--------

Kazu is:

1) An attempt to wrap a curated set of the best open source BioNLP components from the community in a consistent and scalable fashion.
2) Extensible, so that integrating new components should be relatively easy.
3) Configurable - since a Kazu pipeline may have many steps, and each step might have a plethora of configuration options, we manage this complexity by using
   `Hydra <https://hydra.cc/docs/intro/>`_.
4) In production! We're already using Kazu in live drug discovery/development projects, such as the `BIKG knowledge graph <https://www.biorxiv.org/content/10.1101/2021.10.28.466262v1.full>`_.
5) Open source under a permissive Apache 2.0 license - including allowing commercial use.

Kazu is not:

1) a system to train custom models (although we include some TinyBERN2 training code for historical purposes).
