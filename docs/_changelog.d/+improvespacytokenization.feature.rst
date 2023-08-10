Improved spacy tokenization for the ExplosionStringMatchingStep.
Previously, this caused us to miss entities that ended with a single-letter uppercase token at the end (like 'Haemophilia A') if it was at the end of a sentence.
