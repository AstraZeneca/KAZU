FROM rayproject/ray:1.6.0-py38-cpu
COPY  . .
RUN $HOME/anaconda3/bin/pip install .
RUN conda install -c pytorch faiss-cpu
RUN apt-get update
RUN apt-get install unzip
EXPOSE 8000
ENTRYPOINT ["/bin/bash", "scripts/docker/start.sh"]