FROM rayproject/ray:1.6.0-py38-cpu
COPY  . .
RUN $HOME/anaconda3/bin/pip install .
EXPOSE 8000
ENTRYPOINT ["/bin/bash", "scripts/docker/start.sh"]