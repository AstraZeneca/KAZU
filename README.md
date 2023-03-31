![Maturity level-1](https://img.shields.io/badge/Maturity%20Level-ML--1-yellow)

<p align="center">
  <img src="https://raw.githubusercontent.com/AstraZeneca/KAZU/main/docs/kazu_logo.png" alt="Kazu - Biomedical NLP Framework" align=middle style="width: 66%;height: auto;"/>
  <br><br>
</p>

# Kazu - Biomedical NLP Framework

Welcome to Kazu (Korea AstraZeneca University), a python biomedical NLP framework built in collaboration with Korea University,
designed to handle production workloads.

This library aims to simplify the process of using state of the art NLP research in production systems. Some of the 
research contained within are our own, but most of it comes from the community, for which we are immensely grateful.

If you want to use Kazu, please cite our [EMNLP 2022 publication](https://aclanthology.org/2022.emnlp-industry.63)!
([**citation link**](https://aclanthology.org/2022.emnlp-industry.63.bib))

[Please click here for the **web live demo** (Swagger UI) from http://kazu.korea.ac.kr/](http://kazu.korea.ac.kr/)

[Please click here for the TinyBERN2 training and evaluation code](https://github.com/dmis-lab/KAZU-NER-module)


# Quickstart

## Install

Python version 3.9 or higher is required (tested with Python 3.9).

Either: 

`pip install kazu`

or download the wheel from the release page and install locally.

If you intend to use [Mypy](https://mypy.readthedocs.io/en/stable/#) on your own codebase, consider installing Kazu using:

`pip install kazu[typed]`

This will pull in typing stubs for kazu's dependencies (such as [types-requests](https://pypi.org/project/types-requests/) for [Requests](https://requests.readthedocs.io/en/latest/))
so that mypy has access to as much relevant typing information as possible when type checking your codebase. Otherwise (depending on mypy config), you may see errors when running mypy like:

```
.venv/lib/python3.10/site-packages/kazu/steps/linking/post_processing/xref_manager.py:10: error: Library stubs not installed for "requests" [import] 
```

## Getting the model pack

For most functionality, you will also need the Kazu model pack. This is tied to each release, and can be found on the [release page](https://github.com/astrazeneca/kazu/releases). Once downloaded,
extract the archive and:

`export KAZU_MODEL_PACK=<path to the extracted archive>`

Kazu is highly configurable (using [Hydra](https://hydra.cc/docs/intro/)), although it comes preconfigured with defaults appropriate for most literature processing use cases. 
To make use of these, and process a simple document:

```python
import hydra
from hydra.utils import instantiate

from kazu.data.data import Document
from kazu.pipeline import Pipeline
from kazu.utils.constants import HYDRA_VERSION_BASE
from pathlib import Path
import os

# the hydra config is kept in the model pack
cdir = Path(os.environ["KAZU_MODEL_PACK"]).joinpath("conf")


@hydra.main(
    version_base=HYDRA_VERSION_BASE, config_path=str(cdir), config_name="config"
)
def kazu_test(cfg):
    pipeline: Pipeline = instantiate(cfg.Pipeline)
    text = "EGFR mutations are often implicated in lung cancer"
    doc = Document.create_simple_document(text)
    pipeline([doc])
    print(f"{doc.get_entities()}")


if __name__ == "__main__":
    kazu_test()
```

# Documentation

[Find our docs here](https://astrazeneca.github.io/KAZU/_build/html/index.html)

## License

Licensed under [Apache 2.0](https://github.com/AstraZeneca/KAZU/blob/main/LICENSE).

Kazu includes elements under compatible licenses (full licenses are in relevant files or as indicated):
- Some elements are a modification of code licensed under MIT by Explosion.AI - see the README [here](https://github.com/AstraZeneca/KAZU/blob/main/kazu/modelling/ontology_matching/README.md).
- The doc build process (conf.py's linkcode_resolve function) uses code modified from pandas, in turn modified from numpy. See [PANDAS_LICENSE.txt](https://github.com/AstraZeneca/KAZU/blob/main/docs/PANDAS_LICENSE.txt) and [NUMPY_LICENSE.txt](https://github.com/AstraZeneca/KAZU/blob/main/docs/NUMPY_LICENSE.txt)
- Elements of the model distillation code are inspired by or modified from Huawei Noah's Ark Lab [TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/TinyBERT) and DMIS-Lab's [BioBERT](https://github.com/dmis-lab/biobert/tree/master).
  See the details in dataprocessor.py, models.py and tiny_transformer.py.
- PLSapbertModel is inspired by the code from [sapbert](https://github.com/cambridgeltl/sapbert), licensed under MIT. See the file for details, and see the [SapBert](#sapbert) section below regarding use of the model.
- GildaUtils in the string_normalizer.py file is modified from [Gilda](https://github.com/indralab/gilda). See the file for full details
  including the full BSD 2-Clause license.
- The AbbreviationFinderStep uses KazuAbbreviationDetector, which is a modified version of
  [SciSpacy](https://allenai.github.io/scispacy/)'s abbreviation finding algorithm, licensed under Apache 2.0 - see the files for full details.
- The JWTAuthenticationBackend Starlette Middleware in jwtauth.py is originally from [starlette-jwt](https://raw.githubusercontent.com/amitripshtos/starlette-jwt/master/starlette_jwt/middleware.py), licensed under BSD 3-Clause.
- The AddRequestIdMiddleware Starlette Middleware in req_id_header.py is modified from 'CustomHeaderMiddleware' in the [Starlette Middleware docs](https://www.starlette.io/middleware/#basehttpmiddleware).
  This is licensed under BSD 3-Clause along with the rest of Starlette.
- The kazu-jvm folder includes files like gradelw and gradelw.bat distributed by gradle under Apache 2.0 - see the files for details.
- [kazu/data/data.py](https://github.com/AstraZeneca/KAZU/blob/main/kazu/data/data.py) contains `AutoNameEnum`, which is `AutoName` from
  the [Python Enum Docs](https://docs.python.org/3/howto/enum.html#using-automatic-values) licensed under [Zero-Clause BSD](https://docs.python.org/3/license.html#zero-clause-bsd-license-for-code-in-the-python-release-documentation).

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

#### SETH

Kazu's SethStep uses Py4j to call the SETH mutation finder.

Thomas, P., Rocktäschel, T., Hakenberg, J., Mayer, L., and Leser, U. (2016).
[SETH detects and normalizes genetic variants in text](https://pubmed.ncbi.nlm.nih.gov/27256315/)
Bioinformatics (2016)
http://dx.doi.org/10.1093/bioinformatics/btw234
