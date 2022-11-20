from typing import List

from kazu.data.data import Document, CharSpan
from kazu.steps.base.step import Step, iterating_step
from kazu.utils.stanza_pipeline import StanzaPipeline
from stanza.models.common.doc import Sentence


class StanzaStep(Step):
    """
    Stanza step
    Currently used for just sentence-segmentation using a tokenizer trained on the genia treebank

    @inproceedings{qi2020stanza,
        author = {Qi, Peng and Zhang, Yuhao and Zhang, Yuhui and Bolton, Jason and Manning, Christopher D.},
        booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System
        Demonstrations},
        title = {Stanza: A {Python} Natural Language Processing Toolkit for Many Human Languages},
        url = {https://nlp.stanford.edu/pubs/qi2020stanza.pdf},
        year = {2020}
    }

    @Article{10.1093/jamia/ocab090,
        author = {Zhang, Yuhao and Zhang, Yuhui and Qi, Peng and Manning, Christopher D and Langlotz, Curtis P},
        title = "{Biomedical and clinical English model packages for the Stanza Python NLP library}",
        journal = {Journal of the American Medical Informatics Association},
        volume = {28},
        number = {9},
        pages = {1892-1899},
        year = {2021},
        month = {06},
        abstract = "{The study sought to develop and evaluate neural natural language processing (NLP) packages for the
        syntactic analysis and named entity recognition of biomedical and clinical English text.We implement and train
        biomedical and clinical English NLP pipelines by extending the widely used Stanza library originally designed
        for general NLP tasks. Our models are trained with a mix of public datasets such as the CRAFT treebank as well
        as with a private corpus of radiology reports annotated with 5 radiology-domain entities. The resulting
        pipelines are fully based on neural networks, and are able to perform tokenization, part-of-speech tagging,
        lemmatization, dependency parsing, and named entity recognition for both biomedical and clinical text. We
        compare our systems against popular open-source NLP libraries such as CoreNLP and scispaCy, state-of-the-art
        models such as the BioBERT models, and winning systems from the BioNLP CRAFT shared task.For syntactic
        analysis, our systems achieve much better performance compared with the released scispaCy models and CoreNLP
        models retrained on the same treebanks, and are on par with the winning system from the CRAFT shared task. For
        NER, our systems substantially outperform scispaCy, and are better or on par with the state-of-the-art
        performance from BioBERT, while being much more computationally efficient.We introduce biomedical and clinical
        NLP packages built for the Stanza library. These packages offer performance that is similar to the state of the
        art, and are also optimized for ease of use. To facilitate research, we make all our models publicly available.
        We also provide an online demonstration (http://stanza.run/bio).}",
        issn = {1527-974X},
        doi = {10.1093/jamia/ocab090},
        url = {https://doi.org/10.1093/jamia/ocab090},
        eprint = {https://academic.oup.com/jamia/article-pdf/28/9/1892/39731803/ocab090.pdf},
    }
    """

    def __init__(self, stanza_pipeline: StanzaPipeline):
        """

        :param stanza_pipeline: singleton wrapping a stanza pipeline
        """
        self.stanza_nlp = stanza_pipeline.instance

    @iterating_step
    def __call__(self, doc: Document):
        for section in doc.sections:
            stanza_doc = self.stanza_nlp(section.get_text())
            sentences: List[Sentence] = stanza_doc.sentences
            char_spans = (
                CharSpan(sent.tokens[0].start_char, sent.tokens[-1].end_char) for sent in sentences
            )
            section.sentence_spans = char_spans
