FROM ubuntu:18.04 AS downloader

WORKDIR /root/

RUN apt-get update && \
    apt-get install -y  wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh


FROM ubuntu:18.04 AS conda

WORKDIR /root/

COPY Conda-Activator .

COPY --from=downloader /root/Miniconda3-latest-Linux-x86_64.sh .
 
RUN cat Conda-Activator>>/root/.bashrc && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p

RUN rm -f Miniconda3-latest-Linux-x86_64.sh


FROM ubuntu:18.04 AS keras

WORKDIR /root/

COPY --from=conda /root/ /root/
#RUN /bin/sh && /bin/bash 
#RUN conda install -y keras
