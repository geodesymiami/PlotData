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

RUN conda run -n plotdata pip install -r /tmp/requirements.txt
RUN conda run -n plotdata pip install git+https://github.com/geodesymiami/PlotData.git@dev-saveAxis
# RUN conda run -n plotdata pip install MinsarPlotData

ENV PATH=/opt/conda/envs/plotdata/bin:$PATH

ARG SCRATCHDIR
ENV SCRATCHDIR=${SCRATCHDIR}

COPY scripts/check_scratchdir.sh /usr/local/bin/check_scratchdir.sh
RUN chmod +x /usr/local/bin/check_scratchdir.sh

SHELL ["conda", "run", "-n", "plotdata", "/bin/bash", "-c"]

CMD ["bash", "-c", "/usr/local/bin/check_scratchdir.sh && source /tmp/scratchdir.env && plotdata"]