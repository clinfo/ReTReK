FROM ubuntu:18.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update -y

RUN apt-get install -y wget build-essential git

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

RUN mkdir -p /retrek
WORKDIR /retrek
# Installing kMol
RUN git clone https://github.com/elix-tech/kmol.git ~/kmol

RUN conda env create -f ~/kmol/environment.yml
ENV CONDA_DEFAULT_ENV kmol
RUN conda init bash
RUN echo "conda activate kmol" >> ~/.bashrc

# installation of torch-geometric
SHELL ["conda", "run", "-n", "kmol", "/bin/bash", "-c"]
RUN cd ~/kmol && bash install.sh

ENTRYPOINT ["/root/miniconda3/envs/kmol/bin/python"]