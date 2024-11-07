FROM ubuntu:20.04

RUN apt-get update \
  && DEBIAN_FRONTEND="noninteractive" apt-get install -y \
      git \
      python3 \
      xz-utils \
      bzip2 \
      zip \
      gpg \
      wget \
      gpgconf \
      libssl-dev \
      software-properties-common \
      python3-wget \
      python3-bs4 \
      zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

# Cmake ppa
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
RUN echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null

# gcc ppa
RUN add-apt-repository ppa:ubuntu-toolchain-r/test

RUN apt-get update \
  && apt-get install -y \
     gcc-11 \
     g++-11 \
     gfortran-11 \
     cmake \
  && rm -rf /var/lib/apt/lists/*

# mpich
RUN apt-get update \
  && apt-get install -y \
     mpich \
     libmpich-dev \
  && rm -rf /var/lib/apt/lists/*

# Now we install spack and find compilers/externals
RUN mkdir -p /opt/spack && cd /opt/spack && \
  git init && \
  git remote add origin https://github.com/spack/spack.git && \
  git fetch origin 0a4563fd0253c829fb14fd576ad0368954028d5d && \
  git checkout FETCH_HEAD
ADD ./ci/packages.yaml /opt/spack/etc/spack/packages.yaml
RUN . /opt/spack/share/spack/setup-env.sh && spack compiler find
RUN . /opt/spack/share/spack/setup-env.sh && spack external find --not-buildable && spack external list
RUN . /opt/spack/share/spack/setup-env.sh && spack mirror add spack-build-cache-v0.20 https://binaries.spack.io/v0.20.0 && spack buildcache keys --install --trust

ADD ./ci/spack.yaml /opt/spack-environment/spack.yaml
RUN cd /opt/spack-environment \
  && . /opt/spack/share/spack/setup-env.sh \
  && spack env activate . \
  && spack concretize -f

RUN cd /opt/spack-environment \
  && . /opt/spack/share/spack/setup-env.sh \
  && spack env activate . \
  && spack install --show-log-on-error --fail-fast \
  && spack gc -y

# We need to build a specific branch of VeloC until https://github.com/ECP-VeloC/VELOC/pull/43 is resolved
RUN mkdir -p /opt/build/ && cd /opt/build/ && git clone --depth 1 --branch "add-cmake-config-support" https://github.com/nmm0/VELOC.git
RUN mkdir -p /opt/veloc && cd /opt/build/VELOC && python3 ./auto-install.py --without-boost /opt/veloc '"-DBoost_ROOT=/opt/view/gcc-11.4.0/boost/1.81.0"'
RUN rm -rf /opt/build
ENV LD_LIBRARY_PATH /opt/veloc/lib:$LD_LIBRARY_PATH
