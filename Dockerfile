FROM continuumio/miniconda3:latest

RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict

RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml /tmp/environment.yml

RUN conda env create -f /tmp/environment.yml --verbose

COPY requirements.txt /tmp/requirements.txt

RUN conda run -n geo_env pip install -r /tmp/requirements.txt
RUN conda run -n geo_env pip install MinsarPlotData

ENV PATH=/opt/conda/envs/geo_env/bin:$PATH

ARG SCRATCHDIR
ENV SCRATCHDIR=${SCRATCHDIR}

COPY scripts/check_scratchdir.sh /usr/local/bin/check_scratchdir.sh
RUN chmod +x /usr/local/bin/check_scratchdir.sh

SHELL ["conda", "run", "-n", "geo_env", "/bin/bash", "-c"]

CMD ["bash", "-c", "/usr/local/bin/check_scratchdir.sh && source /tmp/scratchdir.env && plotdata"]