#!/usr/bin/env bash

module purge
module use /apps/easybuild/2025/core/modules/all
module use /apps/easybuild/2025/cuda/modules/all
module load compiler/GCC
module load tools/Meson
module load tools/Ninja
module load system/CUDA

export http_proxy=http://webproxy.zdv.uni-mainz.de:8888
export https_proxy=$http_proxy

export CPM_SOURCE_CACHE=$HOME/cpm-cache