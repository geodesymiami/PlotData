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
RUN git clone --branch dev-saveAxis https://github.com/geodesymiami/PlotData.git /tmp/PlotData && \
    cd /tmp/PlotData && \
    echo "Current branch:" && git rev-parse --abbrev-ref HEAD && \
    echo "Latest commit:" && git log -1 && \
    sleep 10 && \
    conda run -n plotdata pip install .
# RUN conda run -n plotdata pip install MinsarPlotData

ENV PATH=/opt/conda/envs/plotdata/bin:$PATH

RUN apt-get update && apt-get install -y \
    xvfb \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

ARG SCRATCHDIR
ENV SCRATCHDIR=${SCRATCHDIR}

COPY scripts/check_scratchdir.sh /usr/local/bin/check_scratchdir.sh
RUN chmod +x /usr/local/bin/check_scratchdir.sh

SHELL ["conda", "run", "-n", "plotdata", "/bin/bash", "-c"]

CMD ["bash", "-c", "Xvfb :99 -screen 0 1024x768x16 & export DISPLAY=:99 && /usr/local/bin/check_scratchdir.sh && source /tmp/scratchdir.env && plotdata"]
# CMD ["bash", "-c", "/usr/local/bin/check_scratchdir.sh && source /tmp/scratchdir.env && plotdata"]