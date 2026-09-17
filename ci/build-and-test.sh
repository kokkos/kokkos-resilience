#!/bin/bash

set -x

cmake --preset $@
cmake --build --preset $@
ctest --preset $@
