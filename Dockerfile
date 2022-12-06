FROM ubuntu:18.04

# from https://fabiorosado.dev/blog/install-conda-in-docker/
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# also from https://fabiorosado.dev/blog/install-conda-in-docker/
ENV CONDA_DIR /opt/conda

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR /tmp

COPY ./env.yml /tmp

RUN conda env create --name rcgan-env -f env.yml

RUN rm /tmp/env.yml

RUN echo "conda activate rcgan-env" > ~/.bashrc

WORKDIR /app

COPY . .

RUN ls -lrt
