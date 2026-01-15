# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# ----------------------------------------------------------------------------
# If you submit this package back to Spack as a pull request,
# please first remove this boilerplate and all FIXME comments.
#
# This is a template package file for Spack.  We've put "FIXME"
# next to all the things you'll want to change. Once you've handled
# them, you can save this file and test your package like this:
#
#     spack install kokkos-resilience
#
# You can edit this file again by typing:
#
#     spack edit kokkos-resilience
#
# See the Spack documentation for more information on packaging.
# ----------------------------------------------------------------------------

from spack_repo.builtin.build_systems.cmake import CMakePackage
from spack.package import *


class KokkosResilience(CMakePackage):
    """FIXME: Put a proper description of your package here."""

    # FIXME: Add a proper url for your package's homepage here.
    homepage = "https://github.com/kokkos/kokkos-resilience"
    git = "https://github.com/kokkos/kokkos-resilience.git"

    # FIXME: Add a list of GitHub accounts to
    # notify when the package is updated.
    # maintainers("github_user1", "github_user2")

    # FIXME: Add the SPDX identifier of the project's license below.
    # See https://spdx.org/licenses/ for a list.
    license("UNKNOWN")

    version("main")

    variant("tracing", default=False, description="Enable tracing of resilience functions")

    variant("automatic", default=True,
        description="Compile automatic checkpointing contexts and backends")

    all_contexts = ["mpi"]
    variant("context", when="+automatic",
        values=any_combination_of(*all_contexts).with_default("mpi"),
        description="Build with automatic checkpointing contexts enabled"
    )

    all_backends = ["veloc"]
    variant("backend", when="+automatic",
        values=any_combination_of(*all_backends).with_default("veloc"),
        description="Build with automatic checkpointing backends enabled"
    )

    all_data_spaces = ["stdfile", "hdf5"]
    variant("data_space",
        values=any_combination_of(*all_data_spaces).with_default("none"),
        description="Build with resilient data spaces enabled"
    )
    variant("hdf5_parallel", default=False, when="data_space=hdf5",
        description="Use parallel version of HDF5")

    all_exec_spaces = ["openmp", "cuda"]
    variant("exec_space", 
        values=any_combination_of(*all_exec_spaces).with_default("none"),
        description="Build with resilient execution spaces enabled"
    )


    depends_on("cmake@3.17:")
    depends_on("kokkos@4.0.01:")
    depends_on("kokkos@4.0.01: +openmp", when="exec_space=openmp")
    depends_on("kokkos@4.0.01: +cuda",   when="exec_space=cuda")
    depends_on("hdf5", when="data_space=hdf5")
    depends_on("boost@1.81.0:", type="build")
    depends_on("mpi", when="backend=veloc")
    depends_on("mpi", when="+hdf5_parallel")
    depends_on("veloc@1.7:", when="backend=veloc")

    def variant_to_args(self, category, values):
        return [
            self.define(
                "KR_ENABLE_" + val.upper() + "_" + category.upper(),
                self.spec.satisfies(category + "=" + val)
            ) for val in values
        ]

    def cmake_args(self):
        args = [
            self.define("KR_ENABLE_TESTS", self.run_tests),
            self.define("KR_ENABLE_EXAMPLES", self.run_tests),
            self.define_from_variant("KR_ENABLE_TRACING", "tracing"),
            self.define_from_variant("KR_ENABLE_AUTOMATIC_CHECKPOINTING", "automatic"),
            self.define("KR_ENABLE_EXEC_SPACES", not self.spec.satisfies("exec_space=none")),
            self.define("KR_ENABLE_DATA_SPACES", not self.spec.satisfies("data_space=none"))
        ]
        
        args.extend(self.variant_to_args("context", self.all_contexts))
        args.extend(self.variant_to_args("backend", self.all_backends))
        args.extend(self.variant_to_args("data_space", self.all_data_spaces))
        args.extend(self.variant_to_args("exec_space", self.all_exec_spaces))

        return args
