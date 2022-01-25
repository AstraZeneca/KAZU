#!/bin/bash

#download the model pack using Azure SAS
echo "downloading model pack"
mkdir -p model_pack
wget --progress=bar:force:noscroll "$MODEL_PACK" -O model_pack/model_pack.zip
#unpack the models
echo "unzipping models"
cd model_pack
unzip  model_pack.zip
mv model_pack/* .
export KAZU_MODEL_PACK=/home/ray/model_pack/
cd ..
echo "running ray deployment"
RAY_SERVE_CONFIG="${RAY_SERVE_CONFIG:-local}"
echo "ray config set to ${RAY_SERVE_CONFIG}"
export TOKENIZERS_PARALLELISM=false
python -m kazu.web.server \
ray=${RAY_SERVE_CONFIG} \
hydra.run.dir="."
echo "ray deployment complete"
