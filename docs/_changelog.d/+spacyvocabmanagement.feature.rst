New SpacyPipelines abstraction, which allows using the same spacy pipeline in different places, but only load it once and prevent uncontrolled memory growth.
On the uncontrolled memory growth, see https://github.com/explosion/spaCy/discussions/10015 for why this was happening - the 'fix' is to reload a spacy pipeline after a certain number of calls.
