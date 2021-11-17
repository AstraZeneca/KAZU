#!/bin/bash

#download the model pack using Azure SAS
echo "downloading model pack"
wget $MODEL_PACK .
#unpack the models
echo "unzipping models"
tar -xf model_pack.tar.gz
echo "running ray deployment"
python -m azner.web.server \
SapBertForEntityLinkingStep=docker \
ray=cluster \
TransformersModelForTokenClassificationNerStep=docker
echo "ray deployment complete"
