FROM rayproject/ray:1.6.0-py38-cpu
COPY  . .
RUN sudo apt-get update
RUN sudo apt-get install unzip
RUN conda install -c pytorch faiss-cpu
RUN $HOME/anaconda3/bin/pip install .
EXPOSE 8000
ENTRYPOINT ["/bin/bash", "scripts/docker/start.sh"]