#!/bin/bash

set -x

git clone --branch feature/cpp --depth 1 https://github.com/sandialabs/Fenix.git /home/kruser/fenix-src
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/opt/view/gcc-11.4.0/openmpi/5.0.5/bin/mpicc -DCMAKE_CXX_COMPILER=/opt/view/gcc-11.4.0/openmpi/5.0.5/bin/mpicxx -DCMAKE_INSTALL_PREFIX=/home/kruser/fenix -B /home/kruser/fenix-build -S /home/kruser/fenix-src
cmake --build /home/kruser/fenix-build --target all
cmake --build /home/kruser/fenix-build --target install

cd /home/kruser/kokkos-resilience-src
cmake --preset ci
cmake --build --preset ci
ctest --preset ci
