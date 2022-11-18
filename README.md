![Maturity level-1](https://img.shields.io/badge/Maturity%20Level-ML--1-yellow)

![Alt text](docs/kazu_logo.png?raw=true "Kazu - Biomedical NLP Framework")

# Kazu - Biomedical NLP Framework

Welcome to Kazu (Korea AstraZeneca University), a python biomedical NLP framework built in collaboration with Korea University,
designed to handle production workloads.

This library aims to simplify the process of using state of the art NLP research in production systems. Some of the 
research contained within are our own, but most of it comes from the community, for which we are immensely grateful.

If you want to use Kazu, please cite our EMNLP 2022 publication!

citation link TBA

[Please click here for the TinyBERN2 training and evaluation code](https://github.com/dmis-lab/KAZU-NER-module)


# Quickstart

## Install

Either: 

`pip install kazu`

or download the wheel from the release page and install locally.

## Getting the model pack

For most functionality, you will also need the Kazu model pack. This is tied to each release, and can be found on the [release page](https://github.com/astrazeneca/kazu/releases). Once downloaded,
extract the archive and:

`export KAZU_MODEL_PACK=<path to the extracted archive>`

Kazu is highly configurable (using [Hydra](https://hydra.cc/docs/intro/)), although it comes preconfigured with defaults appropriate for most literature processing use cases. 
To make use of these, and process a simple document:

```python

from hydra import initialize_config_dir, compose

from kazu.data.data import Document
from kazu.pipeline import Pipeline, load_steps
from pathlib import Path
import os

# the hydra config is kept in the model pack
cdir = Path(os.environ["KAZU_MODEL_PACK"]).joinpath('conf')  
with initialize_config_dir(config_dir=str(cdir)):
    cfg = compose(
        config_name="config",
        overrides=[],
    )
    pipeline = Pipeline(load_steps(cfg))
    text = "EGFR mutations are often implicated in lung cancer"
    doc = Document.create_simple_document(text)
    pipeline([doc])
    print(f"{doc.get_entities()}")

```

# Documentation

[Find our docs here](https://psychic-chainsaw-f197cc2b.pages.github.io/_build/html/index.html)

## License

Licensed under [Apache 2.0](LICENSE).

Kazu includes elements under compatible licenses:
- some elements are a modification of code licensed under MIT by Explosion.AI - see the README [here](kazu/modelling/ontology_matching/README.md).
- the doc build process (conf.py's linkcode_resolve function) uses code modified from pandas, in turn modified from numpy. See [PANDAS_LICENSE.txt](docs/PANDAS_LICENSE.txt) and [NUMPY_LICENSE.txt](docs/NUMPY_LICENSE.txt)

## Dataset licences

### Under [Creative Commons Attribution-Share Alike 3.0 Unported Licence](https://creativecommons.org/licenses/by/3.0/legalcode)

#### Chembl
ChEMBL data is from http://www.ebi.ac.uk/chembl - the version of ChEMBL is ChEMBL_29

#### CLO
CLO data is from http://www.ebi.ac.uk/ols/ontologies/clo - downloaded 18th October 2021

#### UBERON
UBERON data is from http://www.ebi.ac.uk/ols/ontologies/uberon - downloaded 18th October 2021


### Under [Creative Commons Attribution 4.0 Unported License](https://creativecommons.org/licenses/by/4.0/legalcode>)

#### MONDO
MONDO data is from http://www.ebi.ac.uk/ols/ontologies/mondo - downloaded 29th July 2022

#### CELLOSAURUS
CELLOSAURUS data is from https://www.cellosaurus.org/ - downloaded 8th November 2021

#### Gene Ontology
Gene Ontology data is from (version https://zenodo.org/record/7186998#.Y2OcR-zP3iM )


### Other licenced datasets and models

#### OPEN TARGETS
Open Targets datasets are kindly provided by www.opentargets.org, which are free for commercial use cases <https://platform-docs.opentargets.org/licence>

Ochoa, D. et al. (2021). Open Targets Platform: supporting systematic drug–target identification and prioritisation. Nucleic Acids Research.
https://doi.org/10.1093/nar/gkaa1027

#### STANZA

The Stanza framework:

Peng Qi, Yuhao Zhang, Yuhui Zhang, Jason Bolton and Christopher D. Manning. 2020. Stanza: A Python Natural Language Processing Toolkit for Many Human Languages. In Association for Computational Linguistics (ACL) System Demonstrations. 2020.
https://arxiv.org/abs/2003.07082

Biomedical NLP models are derived from: 

Yuhao Zhang, Yuhui Zhang, Peng Qi, Christopher D. Manning, Curtis P. Langlotz. 
Biomedical and Clinical English Model Packages in the Stanza Python NLP Library, 
Journal of the American Medical Informatics Association. 2021.
https://doi.org/10.1093/jamia/ocab090


#### SCISPACY

Biomedical scispacy models are derived from

Mark Neumann, Daniel King, Iz Beltagy, Waleed Ammar
ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing
Proceedings of the 18th BioNLP Workshop and Shared Task
ACL 2019
https://www.aclweb.org/anthology/W19-5034

#### SAPBERT

Kazu uses a distilled form of SAPBERT, from

Fangyu Liu, Ehsan Shareghi, Zaiqiao Meng, Marco Basaldella, Nigel Collier
Self-Alignment Pretraining for Biomedical Entity Representations
ACL 2021
https://aclanthology.org/2021.naacl-main.334/
