# Base image
FROM nvidia/cudagl:10.2-devel-ubuntu18.04

# Install cudnn
ENV CUDNN_VERSION 7.6.5.32
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

# Setup basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libegl1-mesa-dev \
    pkg-config \
    wget \
    zip \
    unzip \
    libcudnn7=$CUDNN_VERSION-1+cuda10.2 \
    libcudnn7-dev=$CUDNN_VERSION-1+cuda10.2 \
    && \
    apt-mark hold libcudnn7 &&\
    rm -rf /var/lib/apt/lists/*

# Install a few libraries to support both EGL and OSMESA options
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    wget \
    doxygen \
    curl \
    libjsoncpp-dev \
    libepoxy-dev \
    libglm-dev \
    libosmesa6 \
    libosmesa6-dev \
    libglew-dev \
    libopencv-dev \
    python-opencv

# Install conda
RUN curl -o ~/miniconda.sh -LO  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    chmod +x ~/miniconda.sh &&\
    ~/miniconda.sh -b -p /opt/conda &&\
    rm ~/miniconda.sh &&\
    /opt/conda/bin/conda install \
        pyyaml \
        mkl \
        mkl-include &&\
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# Install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.13.4/cmake-3.13.4-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.13.4-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

# Conda environment
RUN conda create -n jax python=3.7
RUN /bin/bash -c ". activate jax; pip install --upgrade pip"
RUN /bin/bash -c ". activate jax; conda install numpy scipy h5py ipython"
RUN /bin/bash -c ". activate jax; conda install pandas seaborn jupyterlab matplotlib"
RUN /bin/bash -c ". activate jax; conda install -c conda-forge ipywidgets"

# Build Jax
# install jaxlib
RUN /bin/bash -c ". activate jax; pip install --upgrade https://storage.googleapis.com/jax-releases/cuda102/jaxlib-0.1.47-cp37-none-linux_x86_64.whl; pip install --upgrade jax  # install jax"

# Install tensorflow and tensorflow_datasets
RUN /bin/bash -c ". activate jax; pip install tensorflow tensorflow_datasets"

# Install pytorch + torchvision
RUN /bin/bash -c ". activate jax; pip install torch torchvision"

# Tmux
RUN apt-get install -y tmux
RUN /bin/bash -c "echo $'unbind C-b\n \
set-option -g prefix C-a\n \
bind-key C-a send-prefix\n \
'>/root/.tmux.conf"

# Silence jax logs
ENV GLOG_minloglevel=2
ENV MAGNUM_LOG="quiet"
