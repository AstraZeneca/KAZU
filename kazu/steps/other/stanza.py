try:
    import stanza
    from stanza.models.common.doc import Sentence
except ImportError as e:
    raise ImportError(
        "To use StanzaStep, you need to install stanza.\n"
        "You can either install stanza yourself, or install kazu[all-steps].\n"
    ) from e

from kazu.data import Document, CharSpan
from kazu.steps import Step, document_iterating_step


class StanzaStep(Step):
    """Currently just provides sentence-segmentation using a tokenizer trained on the
    genia treebank.

    .. attention::

        To use this step, you will need `stanza <https://stanfordnlp.github.io/stanza/>`_
        installed, which is not installed as part of the default kazu install
        because this step isn't used as part of the default pipeline.

        You can either do:

        .. code-block:: console

            $ pip install stanza

        Or you can install required dependencies for all steps included in kazu
        with:

        .. code-block:: console

            $ pip install kazu[all-steps]

    Stanza paper:

    Peng Qi, Yuhao Zhang, Yuhui Zhang, Jason Bolton and Christopher D. Manning. 2020.
    `Stanza: A Python Natural Language Processing Toolkit for Many Human Languages. <https://arxiv.org/abs/2003.07082>`_
    In Association for Computational Linguistics (ACL) System Demonstrations. 2020.
    [`pdf <https://nlp.stanford.edu/pubs/qi2020stanza.pdf>`_][`bib <https://nlp.stanford.edu/pubs/qi2020stanza.bib>`_]

    Stanza biomedical and clinical models:

    | Yuhao Zhang, Yuhui Zhang, Peng Qi, Christopher D. Manning, Curtis P. Langlotz.
    | `Biomedical and Clinical English Model Packages in the Stanza Python NLP Library <https://doi.org/10.1093/jamia/ocab090>`_,
    | Journal of the American Medical Informatics Association. 2021.

    .. raw:: html

        <details>
        <summary>Bibtex Citation Details (both papers above)</summary>

    .. code:: bibtex

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

    .. raw:: html

        </details>
    """

    def __init__(self, stanza_pipeline: stanza.Pipeline):
        """

        :param stanza_pipeline: The stanza pipeline the step uses for sentence-segmentation
        """
        self.stanza_pipeline = stanza_pipeline

    @document_iterating_step
    def __call__(self, doc: Document) -> None:
        for section in doc.sections:
            stanza_doc = self.stanza_pipeline(section.text)
            sentences: list[Sentence] = stanza_doc.sentences
            char_spans = (
                CharSpan(sent.tokens[0].start_char, sent.tokens[-1].end_char) for sent in sentences
            )
            section.sentence_spans = char_spans
