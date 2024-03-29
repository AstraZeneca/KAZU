ARG BASE_IMAGE=python:3.9
FROM $BASE_IMAGE

ARG MODEL_PACK_PATH
ARG MODEL_PACK_CONFIG_PATH
ARG TOKEN_GITHUB
ARG ARTIFACTORY_URL
ARG ARTIFACTORY_USER
ARG ARTIFACTORY_TOKEN
ARG KAZU_VERSION
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy

ENV GH_TOKEN $TOKEN_GITHUB

COPY $MODEL_PACK_PATH /model_pack.zip
RUN unzip model_pack.zip -d /model_pack

RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0.0"
RUN pip install --extra-index-url https://$ARTIFACTORY_USER:$ARTIFACTORY_TOKEN@$ARTIFACTORY_URL kazu[webserver]==$KAZU_VERSION

RUN apt-get update
RUN apt-get install -y openjdk-11-jre

ENV KAZU_MODEL_PACK $MODEL_PACK_CONFIG_PATH
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/
ENV HYDRA_FULL_ERROR 1

CMD python -m kazu.web.server --config-path "${KAZU_MODEL_PACK}/conf/" hydra.run.dir="." Middlewares=jwt +ray.init.num_cpus=1 +ray.init.object_store_memory=1000000000
