ARG BASE_IMAGE python:3.9
FROM $BASE_IMAGE

ARG MODEL_PACK_PATH
ARG MODEL_PACK_CONFIG_PATH
ARG KAZU_PACKAGE
ARG TOKEN_GITHUB
ARG ARTIFACTORY_USER
ARG ARTIFACTORY_TOKEN
ARG KAZU_VERSION
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG KAZU_VERSION

ENV GH_TOKEN $TOKEN_GITHUB
ENV KAZU_VERSION $KAZU_VERSION
ENV DEPLOYMENT $DEPLOYMENT

COPY $MODEL_PACK_PATH /model_pack.zip

RUN apt-get update
RUN apt-get install -y openjdk-11-jre

ENV KAZU_MODEL_PACK $MODEL_PACK_CONFIG_PATH
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/
ENV HYDRA_FULL_ERROR 1

COPY $KAZU_PACKAGE .
RUN pip install --upgrade pip
RUN pip install hatch
RUN hatch build
RUN pip install "./dist/kazu-$KAZU_VERSION-py3-none-any.whl[dev]"

RUN unzip model_pack.zip -d /model_pack
CMD python -m kazu.web.server --config-path "${KAZU_MODEL_PACK}/conf/" hydra.run.dir="." Middlewares=jwt +ray.init.num_cpus=1 +ray.init.object_store_memory=1000000000
