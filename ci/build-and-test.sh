#!/bin/bash

set -x

. /opt/spack/share/spack/setup-env.sh

cmake --preset ci
cmake --build --preset ci
ctest --preset ci
