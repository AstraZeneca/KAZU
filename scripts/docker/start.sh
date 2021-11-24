#!/bin/bash

#download the model pack using Azure SAS
echo "downloading model pack"
wget "$MODEL_PACK" -O model_pack.zip
#unpack the models
echo "unzipping models"
unzip  model_pack.zip
echo "running ray deployment"
python -m azner.web.server \
SapBertForEntityLinkingStep=docker \
ray=cluster \
TransformersModelForTokenClassificationNerStep=docker \
DictionaryEntityLinkingStep=docker \
hydra.run.dir="."
echo "ray deployment complete"
