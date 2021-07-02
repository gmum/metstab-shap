FROM continuumio/miniconda3

WORKDIR /app
COPY . .
RUN conda env create -f env/environment.yml
