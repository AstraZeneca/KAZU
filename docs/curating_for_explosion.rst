Curating ontology terms for NER
================================

Many entities in Biomedical NER do not require sophisticated NER or disambiguation techniques, as they are often
unambiguous, and have few genuine synonyms. For instance, terms such as "Breast Cancer" and "mitosis" can be taken at face value, and
simple string matching techniques can be employed (in our case, we use the `spacy PhraseMatcher <https://spacy.io/api/phrasematcher>`_).

However, the terms in ontologies tend to be noisy when taken 'wholesale', and need curation in order to ensure high precision matching.

In Kazu, we take the following approach:

1) generate synonym candidates from the raw ontology to build a putative pipeline

>>> from kazu.modelling.ontology_matching import assemble_pipeline
    from kazu.modelling.ontology_preprocessing.base import MondoOntologyParser
    from kazu.modelling.ontology_preprocessing.synonym_generation import CombinatorialSynonymGenerator,StringReplacement,StopWordRemover


>>> syn_generator = CombinatorialSynonymGenerator([StopWordRemover(),StringReplacement(include_greek=True)])
    parser = MondoOntologyParser(in_path='',data_origin='test',synonym_generator=syn_generator)
    nlp = assemble_pipeline.main(parser_name_to_entity_type={parser.name:'disease'},
                                parsers = [parser],
                                labels = {'disease'},
                                output_dir='~/noisy_spacy_pipeline')

2) we then run this pipeline over a large corpora of text, and look at the frequency of each hit. Note, the below
    is for illustration only - you'll probably want a more sophisticated set up when doing this on a large document set!

>>> from kazu.steps.joint_ner_and_linking.explosion import ExplosionStringMatchingStep
    from dataclasses import dataclass, field
    from typing import List
    import json

    @dataclass
    class AnnotatedPhrase():
        term:str
        action:str
        symbolic:bool
        case_sensitive:bool
        term_norm_mapping:dict[str,str] = field(default_factory=dict)
        examples:list[str] = field(default_factory=list)

    class AnnotatedPhraseEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, AnnotatedPhrase):
                return obj.__dict__
            # Base class default() raises TypeError:
            return json.JSONEncoder.default(self, obj)

    def save(path,data):
        with open(path,'w') as f:
            f.writelines([f"{json.dumps(x,cls=AnnotatedPhraseEncoder)}\n" for x in data])

    noisy_step = ExplosionStringMatchingStep(path='~/noisy_spacy_pipeline')
    docs:List[Document] = get_docs()
    noisy_step(docs)



    curatable_phrases = []
    for doc in docs:
        for section in doc.sections:
            for ent in section.entities:
                term_norm_mapping = {term.parser_name:term.term_norm for term in ent.syn_term_to_synonym_terms}
                symbolic = any(x.is_symbolic for x in ent.syn_term_to_synonym_terms)
                to_curate = AnnotatedPhrase(
                    term = ent.match,
                    action='to_curate',
                    case_sensitive=True,
                    symbolic=symbolic,
                    term_norm_mapping=term_norm_mapping,
                    examples=[section.text[ent.start:ent.end]]
                )
                curatable_phrases.append(to_curate)

    save('~/phrases_to_curate.jsonl')


3) we curate the phrases_to_curate.jsonl file, according to whether they look like good matches or not for a given parser, and whether case matters

4) Now, the final pipeline can be generated as follows:

>>> nlp = assemble_pipeline.main(parser_name_to_entity_type={parser.name:'disease'},
                                    curated_list = '~/phrases_to_curate.jsonl',
                                    labels = {'disease'},
                                    output_dir='~/<kazu model pack>/spacy_pipeline')
