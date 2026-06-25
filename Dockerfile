FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV CONDA_ENV_NAME=hunyuan
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    bzip2 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm -f /tmp/miniconda.sh \
    && "${CONDA_DIR}/bin/conda" clean -afy

RUN "${CONDA_DIR}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && "${CONDA_DIR}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

WORKDIR /workspace/hunyuan3.0_assets_creation

COPY environment.yml /tmp/environment.yml

RUN "${CONDA_DIR}/bin/conda" env create -f /tmp/environment.yml \
    && "${CONDA_DIR}/bin/conda" clean -afy \
    && rm -f /tmp/environment.yml

ENV PATH=${CONDA_DIR}/envs/${CONDA_ENV_NAME}/bin:${CONDA_DIR}/bin:${PATH}

COPY . /workspace/hunyuan3.0_assets_creation

CMD ["bash"]
