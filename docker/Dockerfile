FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV CONDA_ENV_NAME=hunyuan_sam3d
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10
ENV PIP_RESUME_RETRIES=10
ENV PIP_CONFIG_FILE=/dev/null
ENV PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_EXTRA_INDEX_URL=
ENV PIP_FIND_LINKS=
ENV PATH=${CONDA_DIR}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    bzip2 \
    build-essential \
    ca-certificates \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm -f /tmp/miniconda.sh \
    && "${CONDA_DIR}/bin/conda" clean -afy

RUN "${CONDA_DIR}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && "${CONDA_DIR}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

WORKDIR /workspace/hunyuan3.0_assets_creation

COPY environment.yml /tmp/environment.yml

RUN --mount=type=cache,target=/root/.cache/pip \
    "${CONDA_DIR}/bin/conda" env create -f /tmp/environment.yml \
    && "${CONDA_DIR}/bin/conda" clean -afy \
    && rm -f /tmp/environment.yml

ENV PATH=${CONDA_DIR}/envs/${CONDA_ENV_NAME}/bin:${CONDA_DIR}/bin:${PATH}

COPY . /workspace/hunyuan3.0_assets_creation

CMD ["bash"]
