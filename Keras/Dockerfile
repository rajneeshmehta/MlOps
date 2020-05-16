FROM ubuntu:18.04
RUN apt-get update && \
    apt-get install -y  wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
COPY . .
RUN cat Conda-Activator>>/root/.bashrc
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p