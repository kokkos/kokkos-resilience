#!/bin/bash

set -x

export LSAN_OPTIONS=suppressions=/opt/src/kokkos-resilience/ci/lsan.supp

cmake --preset $@
cmake --build --preset $@
ctest --preset $@
