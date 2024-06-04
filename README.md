![Maturity level-1](https://img.shields.io/badge/Maturity%20Level-ML--2-green)

<p align="center">
  <img src="https://raw.githubusercontent.com/AstraZeneca/KAZU/main/docs/kazu_logo.png" alt="Kazu - Biomedical NLP Framework" align=middle style="width: 66%;height: auto;"/>
  <br><br>
</p>

[Find our docs here](https://astrazeneca.github.io/KAZU/index.html)

# Kazu - Biomedical NLP Framework

**Note: the recent 2.0 release has large elements of backwards incompatibility if you are using a custom model pack and curations.**

Welcome to Kazu (Korea AstraZeneca University), a python biomedical NLP framework built in collaboration with Korea University,
designed to handle production workloads.

This library aims to simplify the process of using state of the art NLP research in production systems. Some of the
research contained within are our own, but most of it comes from the community, for which we are immensely grateful.

If you want to use Kazu, please cite our [EMNLP 2022 publication](https://aclanthology.org/2022.emnlp-industry.63)!
([**citation link**](https://aclanthology.org/2022.emnlp-industry.63.bib))

[Please click here for the TinyBERN2 training and evaluation code](https://github.com/dmis-lab/KAZU-NER-module)

# Quickstart

## Install

Python version 3.9 or higher is required (tested with Python 3.11).

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

from kazu.data import Document
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

## License

Licensed under [Apache 2.0](https://github.com/AstraZeneca/KAZU/blob/main/LICENSE).

Kazu includes elements under compatible licenses (full licenses are in relevant files or as indicated):
- Some elements are a modification of code licensed under MIT by Explosion.AI - see the README [here](https://github.com/AstraZeneca/KAZU/blob/main/kazu/ontology_matching/README.md).
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
- [kazu/data.py](https://github.com/AstraZeneca/KAZU/blob/main/kazu/data.py) contains `AutoNameEnum`, which is `AutoName` from
  the [Python Enum Docs](https://docs.python.org/3/howto/enum.html#using-automatic-values) licensed under [Zero-Clause BSD](https://docs.python.org/3/license.html#zero-clause-bsd-license-for-code-in-the-python-release-documentation).

## Dataset licences

For the version of each ontology currently in use, please see the 'data_origin' field in kazu/conf/ontologies

### Under [Creative Commons Attribution-Share Alike 3.0 Unported Licence](https://creativecommons.org/licenses/by/3.0/legalcode)

#### Chembl
ChEMBL data is from http://www.ebi.ac.uk/chembl

#### CLO
CLO data is from http://www.ebi.ac.uk/ols/ontologies/clo

#### UBERON
UBERON data is from http://www.ebi.ac.uk/ols/ontologies/uberon

### Under [Creative Commons Attribution 4.0 Unported License](https://creativecommons.org/licenses/by/4.0/legalcode>)

#### MONDO
MONDO data is from http://www.ebi.ac.uk/ols/ontologies/mondo

#### CELLOSAURUS
CELLOSAURUS data is from https://www.cellosaurus.org/

#### Gene Ontology
Gene Ontology data is from http://purl.obolibrary.org/obo/go.owl


### Other licenced datasets and models

#### HPO

This service/product uses the Human Phenotype Ontology (version information). Find out more at http://www.human-phenotype-ontology.org

Freely licenced under https://hpo.jax.org/app/license

Sebastian Köhler, Michael Gargano, Nicolas Matentzoglu, Leigh C Carmody, David Lewis-Smith,
Nicole A Vasilevsky, Daniel Danis, Ganna Balagura, Gareth Baynam, Amy M Brower,
Tiffany J Callahan, Christopher G Chute, Johanna L Est, Peter D Galer, Shiva Ganesan,
Matthias Griese, Matthias Haimel, Julia Pazmandi, Marc Hanauer, Nomi L Harris,
Michael J Hartnett, Maximilian Hastreiter, Fabian Hauck, Yongqun He, Tim Jeske, Hugh Kearney,
Gerhard Kindle, Christoph Klein, Katrin Knoflach, Roland Krause, David Lagorce, Julie A McMurry,
Jillian A Miller, Monica C Munoz-Torres, Rebecca L Peters, Christina K Rapp, Ana M Rath,
Shahmir A Rind, Avi Z Rosenberg, Michael M Segal, Markus G Seidel, Damian Smedley,
Tomer Talmy, Yarlalu Thomas, Samuel A Wiafe, Julie Xian, Zafer Yüksel, Ingo Helbig,
Christopher J Mungall, Melissa A Haendel, Peter N Robinson,

The Human Phenotype Ontology in 2021,

Nucleic Acids Research, Volume 49, Issue D1, 8 January 2021, Pages D1207–D1217,<br>
https://doi.org/10.1093/nar/gkaa1043


#### OPEN TARGETS
Open Targets datasets are kindly provided by www.opentargets.org, which are free for commercial use cases <https://platform-docs.opentargets.org/licence>

Ochoa, D. et al. (2021). Open Targets Platform: supporting systematic drug–target identification and prioritisation. Nucleic Acids Research.<br>
https://doi.org/10.1093/nar/gkaa1027

#### STANZA

The Stanza framework:

Peng Qi, Yuhao Zhang, Yuhui Zhang, Jason Bolton and Christopher D. Manning. 2020. Stanza: A Python Natural Language Processing Toolkit for Many Human Languages. In Association for Computational Linguistics (ACL) System Demonstrations. 2020.<br>
https://arxiv.org/abs/2003.07082

Biomedical NLP models are derived from:

Yuhao Zhang, Yuhui Zhang, Peng Qi, Christopher D. Manning, Curtis P. Langlotz.<br>
Biomedical and Clinical English Model Packages in the Stanza Python NLP Library,<br>
Journal of the American Medical Informatics Association. 2021.<br>
https://doi.org/10.1093/jamia/ocab090

#### SCISPACY

Biomedical scispacy models are derived from

Mark Neumann, Daniel King, Iz Beltagy, Waleed Ammar<br>
ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing<br>
Proceedings of the 18th BioNLP Workshop and Shared Task<br>
ACL 2019<br>
https://www.aclweb.org/anthology/W19-5034

#### SAPBERT

Kazu uses a distilled form of SAPBERT, from

Fangyu Liu, Ehsan Shareghi, Zaiqiao Meng, Marco Basaldella, Nigel Collier<br>
Self-Alignment Pretraining for Biomedical Entity Representations<br>
ACL 2021<br>
https://aclanthology.org/2021.naacl-main.334/

#### GLINER

GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer.<br>
Urchade Zaratiana, Nadi Tomeh, Pierre Holat, Thierry Charnois<br>
https://arxiv.org/abs/2311.08526

#### SETH

Kazu's SethStep uses Py4j to call the SETH mutation finder.

Thomas, P., Rocktäschel, T., Hakenberg, J., Mayer, L., and Leser, U. (2016).<br>
[SETH detects and normalizes genetic variants in text](https://pubmed.ncbi.nlm.nih.gov/27256315/)<br>
Bioinformatics (2016)<br>
http://dx.doi.org/10.1093/bioinformatics/btw234


#### Opsin

Kazu's OpsinStep uses Py4j to call OPSIN: Open Parser for Systematic IUPAC nomenclature.

Daniel M. Lowe, Peter T. Corbett, Peter Murray-Rust, and Robert C. Glen<br>
Chemical Name to Structure: OPSIN, an Open Source Solution<br>
Journal of Chemical Information and Modeling 2011 51 (3), 739-753<br>
DOI: [10.1021/ci100384d](https://doi.org/10.1021/ci100384d)
