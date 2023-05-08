FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
LABEL Author="Adiren Guo"

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN rm requirements.txt

RUN apt-get update && apt-get install -y \
    tmux