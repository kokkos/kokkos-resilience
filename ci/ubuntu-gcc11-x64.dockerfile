ARG BASE=ubuntu:20.04
FROM $BASE

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
  && rm -rf /var/lib/apt/lists/* \
  && ln -s /usr/lib/x86_64-linux-gnu/libz.so /usr/lib/libz.so # so that spack build finds libz in the expected place

# Cmake ppa
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -c -s) main" | tee /etc/apt/sources.list.d/kitware.list >/dev/null

# gcc ppa
RUN add-apt-repository ppa:ubuntu-toolchain-r/test

RUN apt-get update \
  && DEBIAN_FRONTEND="noninteractive" apt-get install -y \
     gcc-11 \
     g++-11 \
     gfortran-11 \
     cmake \
  && rm -rf /var/lib/apt/lists/*

# Now we install spack and find compilers/externals
RUN mkdir -p /opt/spack/ \
  && cd /opt/spack/ \
  && git init \
  && git remote add origin https://github.com/spack/spack.git \
  && git fetch origin 6f948eb847c46a9caea852d3ffffd9cd4575dacc \
  && git checkout FETCH_HEAD

RUN . /opt/spack/share/spack/setup-env.sh \
  && spack compiler find \
  && spack external find --not-buildable \
  && spack external list \
  && spack mirror add spack-build-cache-v0.20 https://binaries.spack.io/v0.20.0 \
  && spack buildcache keys --install --trust

# ... and setup the spack environment
ADD ./ci/spack.yaml /opt/spack-environment/spack.yaml

RUN . /opt/spack/share/spack/setup-env.sh \
  && spack env activate /opt/spack-environment \
  && spack concretize -f \
  && spack install --show-log-on-error --fail-fast \
  && spack gc -y

