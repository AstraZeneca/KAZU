# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

<!-- towncrier release notes start -->

## 2.3.0 - 2024-12-17

### Features

- Release new multilabel biomedBERT model trained on LLM (Gemini) synthetically generated NER data. The model was trained on over 7000 LLM annoted documents with a total of 295822 samples.
  The model was trained for 21 epochs and achieved an F1 score of 95.6% on a held out test set. (multilabel_bert)
- added multilabel NER training example and config.
- added scaling kazu with Ray docs and example.

### Bugfixes

- Fix issue with TransformersModelForTokenClassificationNerStep when processing large amounts of documents. The fix offloads tensors onto cpu before performin the torch.cat operation which lead to a zero tensor before. (pytorch_memory_issue)


## 2.2.1 - 2024-10-21

### Features

- Update ontologies to later versions (ontology_updates)

### Bugfixes

- Fix synonym generator to only check if strings exist in original synonyms. Update tests (combinatorial_synonym_generator)
- Remove save/reset button not belonging on page 1 (krt)


## 2.2.0 - 2024-09-18

### Features

- New LLMNERStep, for performing NER with LLMs

### Bugfixes

- Fix bug with Chromosome X being converted to Chromosome 10 raised in #42 (chromosomeX)
- Fix pip install command in docs raised in #56 (docs_pip_command)
- Added new multiword AutoCurationAction, and adjusted some curations as per #58.


## 2.1.1 - 2024-07-08


No significant changes.


## 2.1.0 - 2024-07-04


### Features

- Added new Kazu Resource Tool UI to ease the process of updating resources and resource configuration.
- New OntologyDownloader abstraction to assist with resource updating.
- Updated resources for June 2024.


## 2.0.0 - 2024-06-04


### Features

- (De)serialization has been greatly improved, simplified, made correct, and given a slightly more compact serialized representation.
  This does mean there are some small changes in (de)serialization behaviour since the previous release.
- Curation process has been significantly improved and simplified for the end user, including introducing the `AutoCurator` concept to aid in this. This will enable us to build out better documentation and an interactive tool in future releases, which are currently in draft. Overally, this will greatly simplify upgrading ontology versions, adding curations for a new ontology etc.
- Datamodel has been substantially revised in a **backwards incompatible** manner to clear up confusing concepts, fix longstanding issues etc.
- New Zero shot NER model with GLiNER

### Deprecations and Removals

- Remove deprecated `GildaUtils.replace_dashes`. This was superceded by `GildaUtils.split_on_dashes_or_space` and was already deprecated pending removal.
- Remove deprecated `SpacyToKazuObjectMapper`, as this was renamed to `KazuToSpacyObjectMapper`, and the old name already deprecated pending removal.
- Remove deprecated `create_phrasematchers_using_curations` method of `OntologyMatcher`. This was renamed to `create_phrasematchers` and was already deprecated pending removal.
- Rename `Document.json` to `to_json`, and remove optional arguments.
  The previous name was inconsistent with naming on other classes, as the function signature were parallel to `to_json` methods.
  The argument `drop_unmapped_ents` had functionality that was duplicated with `DropUnmappedEntityFilter` within the `CleanupStep`,
  and it made sense to add the `drop_terms` behaviour to a new `LinkingCandidateRemovalCleanupAction` to collect this behaviour together
  and significantly simplify the Document serialization code.
- Rename `ParserActions.from_json` and `GlobalParserActions.from_json` to `from_dict`.
  The previous names were misleading, as the function signature were parallel to the `from_dict` methods on other classes, not to their `from_json` methods.
- Renamed `SynonymDatabase.add` to `SynonymDatabase.add_parser`, for consistency with `MetadataDatabase.add_parser`.


## 1.5.1 - 2024-01-29


### Bugfixes

- Pinned scipy to <1.12.0 due to breaking API change.


## 1.5.0 - 2024-01-19


### Features

- Added new cleanup action: DropMappingsByParserNameRankAction
- Added new disambiguation strategy: PreferNearestEmbeddingToDefaultLabelDisambiguationStrategy.
- DefinedElsewhereInDocumentDisambiguationStrategy has slightly changed, so that it will only return mappings that were found elsewhere in the document, rather than the whole EquivalentIdSet where those ids were contained
- New disambiguation methodology GildaTfIdfDisambiguationStrategy.
- OpenTargetsTargetOntologyParser now has a biotype filter parameter.

### Deprecations and Removals

- Deprecated `GildaUtils.replace_dashes` in favour of `GildaUtils.split_on_dashes_or_space`, as the latter improves efficiency in Kazu.
  `GildaUtils.replace_dashes` will continue to work until kazu 1.6, but using it will produce a `DeprecationWarning`.
  Please [open a GitHub issue](https://github.com/AstraZeneca/KAZU/issues/new) if you wish this to remain.


## 1.4.0 - 2023-12-01


### Features

- Added new curation_report.py to assist in upgrading ontologies between versions
- New disambiguation strategy to prefer mappings that have a default label that matches an entity.
- The OpenTargetsDiseaseOntologyParser has been heavily reworked, so that it uses the therapeutic_area concept to decide what records should be included. This has in turn yielded the subsets: measurement, medical_procedure, biological_process and phenotype. The measurement configuration is currently disabled as it requires heavy curation of the underlying strings. In addition, the OpenTargetsDiseaseOntologyParser now supports a custom ID grouping method, to make use of cross references.

### Bugfixes

- MemoryEfficientStringMatchingStep now only produces a single entity per class where multiple curations exist with different cases.
- Previously, the `tested_dependencies.txt` file in the model packs included an editable install of kazu, which wasn't intended.
  We now exclude kazu from that output.
- Speed up model pack builds for model packs using `ExplosionStringMatchingStep`, by fixing a bug that caused the parsers to be populated twice in this case.

### Deprecations and Removals

- Removed pytorch-lightning as a dependency. The signatures of SapbertStringSimilarityScorer and TransformersModelForTokenClassificationNerStep have changed
- Renamed `create_phrasematchers_using_curations` method of `OntologyMatcher` to `create_phrasematchers`. The old name will continue to work until kazu 1.6, but using it will produce a `DeprecationWarning`.
- `MetadataDatabase.add_parser` now requires an `entity_class`.
  This enables correct string normalisation in the `MappingStep` for the new disambiguation strategy.


## 1.3.2 - 2023-11-21


### Bugfixes

- Hits with scores of 0.0 are no longer returned by DictionaryIndex
- Pin lightning-utilities dependency, a new version of which completely broke the model inference, despite lightning itself being pinned (they didn't pin lightning-utilities appropriately in the version we're using).


## 1.3.1 - 2023-11-15


### Features

- Added methods to dataclasses that allow them to be deserialied from json.

### Deprecations and Removals

- Renamed `SpacyToKazuObjectMapper` to `KazuToSpacyObjectMapper`.
  The old name will continue to work until kazu 1.6, but using it will produce a `DeprecationWarning`
- `RulesBasedEntityClassDisambiguationFilterStep` no longer requires `parsers` or `other_entity_classes`.
  It previously used these to construct the `entity_classes` argument of `KazuToSpacyObjectMapper.__init__`, but now we can just calculate which of these we really need from the class and mention rules passed to `RulesBasedEntityClassDisambiguationFilterStep.__init__`


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
- New SpacyPipelines abstraction, which allows using the same spaCy pipeline in different places, but only load it once and prevent uncontrolled memory growth.
  On the uncontrolled memory growth, see https://github.com/explosion/spaCy/discussions/10015 for why this was happening - the 'fix' is to reload a spaCy pipeline after a certain number of calls.
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

- Improved spaCy tokenization for the ExplosionStringMatchingStep.
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
