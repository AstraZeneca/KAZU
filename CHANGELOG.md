# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

<!-- towncrier release notes start -->

## 1.0.1 - 2023-07-27


### Features

- added more informative logging to build_and_test_model_packs.py #680


### Bugfixes

- capped pytorch to v <2.0.0 #679
- URIs are now stripped in acceptance testing if called from build_and_test_model_packs.py #684


### Deprecations and Removals

- remove Section.get_text and preprocessed_text #683


### Misc

-  #681, #685


## 1.0.0 - 2023-07-21


### Features

- Added support for SKOS-XL ontologies with SKOSXLGraphParser. 
- Allowed running only some steps in a pipeline (including via API). See the Pipeline docs
  (and the OpenAPI spec for the API details of new endpoints). 
- Curations have been significantly overhauled and are now called CuratedTerms.
  They now apply to single parsers, and offer a range of actions affecting linking and NER behaviour. Each CuratedTerm has a single CuratedTermBeheaviour, making them significantly simpler. 
- Move MergeOverlappingEntsStep to after the AbbreviationFinderStep, which improves precision. 
- Moved model training specific dependencies to an optional dependency group.
  In particular, this is of value because the seqeval dependency doesn't distribute
  a wheel, only an sdist in a legacy manner, which broke kazu installation in
  environments requiring proxies. 
- Switch off URI Stripping by default, and enable customizing the behaviour of URI stripping for Mapping ids.
  See the docs for the StripMappingURIsAction class for details. 
- The LabelStudio integration interface was improved and simplified. 
- We now assign a MentionConfidence to each Entity based on the confidence of the NER hit.
  This allows decoupling between specific NER steps and disambiguation strategies, which were previously intertwined.
  This also applies to the CleanupStep which is decoupled from specific NER steps. 
- We now use diskcache to give a simpler interface to elements of kazu that need to be 'built' in advance.
  This particularly benefits the parsers, where it is now easy to use a new ParserDependentStep abstraction
  to ensure appropriate parsers are loading, but to only load parsers once across all steps. 


### Improved Documentation

- We now generate docs for ``__call__`` methods. These are often significant in KAZU,
  and we had the docstrings written in many cases, Sphinx just didn't show them
  in the html. 


### Deprecations and Removals

- The ``ner`` and ``batch_ner`` API endpoints are deprecated and will be removed
  in a future release. ``ner_and_linking`` should be used instead. This is because
  we now have an ``ner_only`` endpoint, so the naming was liable to cause confusion.
  It also simplifies the api, as a single endpoint can handle both a single document or
  multiple. 
- We deleted the 'modelling' element of the module structure, as this didn't add anything semantically, but led to longer imports.
  This does mean any imports from these directories in your code will need to be rewritten - fortunately this is a simple 'find and delete' with 'modelling.' within your imports.


## 0.1.0 - 2023-03-29


### Features

- Kazu frontend can now be served from the Ray HTTP deployment. 
- Upgraded SETH to 1.4.0 . 
- removed CuratedTerm and replaced with more flexible Curation and GlobalAction system


## 0.0.25 - 2023-02-28


### Features

- Added a Kazu web interface for demoing NER results on text snippets. 


### Improved Documentation

- Added link to changelogs on main documentation page. 
- Added towncrier for generating changelogs. 
- Generating and committing changelog during release workflow. 
- Removed previous changelogs file (NEWS.md).
