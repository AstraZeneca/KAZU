FROM $BASE_IMAGE

ARG MODEL_PACK_PATH
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

ENV GH_TOKEN $TOKEN_GITHUB

COPY $KAZU_PACKAGE .
COPY $MODEL_PACK_PATH /model_pack.zip
RUN unzip model_pack.zip -d /model_pack

RUN apt-get update
RUN apt-get install -y openjdk-11-jre

ENV KAZU_MODEL_PACK /model_pack
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/
ENV HYDRA_FULL_ERROR 1

RUN pip install -e kazu[dev]

CMD python -m kazu.web.server --config-path "/model_pack/conf" hydra.run.dir="." Middlewares=jwt
