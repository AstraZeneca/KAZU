FROM rayproject/ray:1.6.0-py38-cpu
COPY  . .
RUN $HOME/anaconda3/bin/pip install  azner/.
EXPOSE 8000