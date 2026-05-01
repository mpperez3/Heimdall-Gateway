# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates apt-transport-https gnupg2 software-properties-common \
    build-essential git curl cmake ninja-build pkg-config ca-certificates \
    libssl-dev wget python3 python3-venv python3-distutils python3-pip && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    python3 -m pip install --upgrade pip setuptools wheel pytest && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

CMD ["/bin/bash"]
