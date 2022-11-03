# Kazu - Biomedical NLP Framework

Welcome to Kazu (Korea AstraZeneca University), a python biomedical NLP framework built in collaboration with Korea University,
designed to handle production workloads.

This library aims to simplify the process of using state of the art NLP research in production systems. Some of the 
research contained within are our own, but most of it comes from the community, for which we are immensely grateful.

See docs at https://psychic-chainsaw-f197cc2b.pages.github.io/_build/html/index.html

# Quickstart

Either: 
`
pip install -u kazu
`
or download the wheel from the release page and install locally.

For most functionality, you will also need the Kazu model pack. This is tied to each release, and can be found on the release page. Once downloaded,
extract the archive and:

`
export KAZU_MODEL_PACK=<path to the extracted archive>
`

Kazu is highly configurable (using [Hydra](https://hydra.cc/docs/intro/)), although is comes preconfigured with defaults appropriate for most literature processing use cases. 
To make use of these, and process a simple document:

```python

from hydra import initialize_config_dir, compose

from kazu.data.data import Document
from kazu.pipeline import Pipeline, load_steps
from pathlib import Path
import os

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
CLO data is from http://www.ebi.ac.uk/clo - downloaded 18th October 2021

#### UBERON
UBERON data is from http://www.ebi.ac.uk/uberon - downloaded 18th October 2021



### Under [Creative Commons Attribution 4.0 Unported License](https://creativecommons.org/licenses/by/4.0/legalcode>)

#### MONDO
MONDO data is from http://www.ebi.ac.uk/mondo - downloaded 29th July 2022

#### CELLOSAURUS
CELLOSAURUS data is from https://www.cellosaurus.org/ - downloaded 8th November 2021

#### Gene Ontology
Gene Ontology data is from (version https://zenodo.org/record/7186998#.Y2OcR-zP3iM )


### Other licenced datasets

#### OPEN TARGETS
Open Targets datasets are kindly provided by www.opentargets.org, which are free for commercial use cases <https://platform-docs.opentargets.org/licence>
