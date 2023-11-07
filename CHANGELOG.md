# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

<!-- towncrier release notes start -->

## 1.3.0 - 2023-11-07


### Features

- CurationProcessor no longer tries to handle curations with INHERIT_FROM_SOURCE_TERM behaviour, as this was causing confusion and conflicts. This is now the responsibility of the caller.
- Updated ontologies for October 2023.

### Bugfixes

- Fixed a bug in MemoryEfficientStringMatchingStep where caseinsensitive overlaps caused ontology info to be lost.


## 1.2.0 - 2023-10-18


### Features

- added two new synonym generation routines, VerbPhraseVariantGenerator and TokenListReplacementGenerator
- synonym generators now cache results, and are thus much faster


## 1.1.2 - 2023-10-11


### Bugfixes

- fixed a deprecated Iterable import for python 3.10 compatibility.
- fixed an extra indent in RulesBasedEntityClassDisambiguationFilterStep that led to inappropriate matcher rules.


## 1.1.1 - 2023-10-10


### Bugfixes

- fixed a bug where steps depending on additional dependencies were imported in `steps.__init__.py`


## 1.1.0 - 2023-10-10


### Features

- A couple of easy, non-behaviour changing performance improvements that on their own sped up Kazu around 10% (but other changes in this release will affect this too, and speedup will be workload dependent)
- Added new OpsinStep which maps IUPAC drug strings to canonical SMILES - see the API docs for details.
  This functionality is currently experimental and may be changed without making a new major release.
  Please [open a GitHub issue](https://github.com/AstraZeneca/KAZU/issues/new) if you wish to use this functionality.
- Ensembl Gene IDs are now grouped by HGNC approved symbols, eliminating disambiguation problems for gene IDs belonging to the same gene.
- Entity produced by TransformersModelForTokenClassificationNerStep but without Mappings will be dropped by default now, in the same way as for other NER steps.
  This was an exception to handle an AstraZeneca internal use case that wanted this different, but it could cause issues with MergeOverlappingEntsStep in some cases,
  so it is safer to have this off by default.
- New SpacyPipelines abstraction, which allows using the same spacy pipeline in different places, but only load it once and prevent uncontrolled memory growth.
  On the uncontrolled memory growth, see https://github.com/explosion/spaCy/discussions/10015 for why this was happening - the 'fix' is to reload a spacy pipeline after a certain number of calls.
- Slimmed down base dependencies by removing dependencies for steps not in the base pipeline.
  These can be added back in manually in user projects, or use the new `kazu[all-steps]` dependency
  group to install dependencies for all steps as before. The docs reflect this, and informative errors
  are raised when trying to use these steps when dependencies aren't installed.
- Very large memory savings from an overhaul of the string matching process.
  The new version should also be faster in general, but the priority was memory rather than speed (since previously, this step accounted for the majority of kazu's memory usage but only a fraction of its runtime)

### Bugfixes

- Curated terms that drop the same normalised version of the term no longer report erroneous warnings.

### Deprecations and Removals

- The API for building custom model packs has changed to be more flexible, and more simple.
  This is a backwards-incompatible change, but we don't currently expect/know of any non-AstraZeneca users of this script, so won't do a major version bump for it.
  Please let us know (in a [GitHub issue](https://github.com/AstraZeneca/KAZU/issues/new)) if you are using this and this change was problematic for you.


## 1.0.3 - 2023-08-15


### Features

- Improved spacy tokenization for the ExplosionStringMatchingStep.
  Previously, this caused us to miss entities that ended with a single-letter uppercase token at the end (like 'Haemophilia A') if it was at the end of a sentence.
- Make SpanFinder return found spans directly, rather than having to access `.closed_spans` after calling, which is easier. Note that `.closed_spans` remains, so this is backwards-compatible.
- Turned on 'strict' mypy checking (with some exceptions as to the exact flags used), and fixed issues that this raised.


### Bugfixes

- Fix incorrect caching behaviour of Index TfidfVectorizer builds.
  This meant they got rebuilt every time, which meant in turn that the cache and therefore the model pack size grew after use.


### Improved Documentation

- Started using docformatter to automatically format docstrings, and tweak minor issues this brought up.
  This will help us comply with PEP257 and be consistent across the codebase.


### Deprecations and Removals

- Removed various pieces of dead code.
  These are very unlikely to have been used by end users, so not deprecating/doing a major version bump.
- Rename Type Alias JsonDictType to JsonEncodable - which is more straightforward/correct what it actually means.
  This was used internally to Kazu rather than being expected to be used by end users, so no deprecation/major version bump.


## 1.0.2 - 2023-08-07


### Bugfixes

- Added upper version limit for ray[serve] for the webserver dependencies.
  In ray 2.5, HTTP Proxy Health checks were introduced which by default kill slow-deploying servers.
  There are environments variables that can override this behaviour, but specifying them at the right time
  is a pain in our setup, so until we've decided on the best way of handling this, just pin to a version of
  ray that works.


## 1.0.1 - 2023-07-31


### Features

- added more informative logging to build_and_test_model_packs.py
- dependencies that work are now stored in the model pack


### Bugfixes

- URIs are now stripped in acceptance testing if called from build_and_test_model_packs.py
- fixed CI bug installing cuda pytorch which we don't want


### Improved Documentation

- added sphinx docs to data classes


### Deprecations and Removals

- remove Section.get_text and preprocessed_text


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
