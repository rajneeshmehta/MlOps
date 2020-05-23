FROM ubuntu:18.04 AS downloader
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 

FROM ubuntu:18.04 As conda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
COPY --from=downloader Miniconda3-latest-Linux-x86_64.sh .
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p && \
    rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda install tensorflow
