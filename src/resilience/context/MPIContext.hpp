/*
 *
 *                        Kokkos v. 3.0
 *       Copyright (2020) National Technology & Engineering
 *               Solutions of Sandia, LLC (NTESS).
 *
 * Under the terms of Contract DE-NA0003525 with NTESS,
 * the U.S. Government retains certain rights in this software.
 *
 * Kokkos is licensed under 3-clause BSD terms of use:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the Corporation nor the names of the
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Questions? Contact Christian R. Trott (crtrott@sandia.gov)
 */
#ifndef INC_KOKKOS_RESILIENCE_MPICONTEXT_HPP
#define INC_KOKKOS_RESILIENCE_MPICONTEXT_HPP

#include <mpi.h>
#include "ContextBase.hpp"

namespace KokkosResilience {

//Helper for initialization list
namespace {
static int comm_rank(MPI_Comm &comm){
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}
}

class MPIContext : public ContextBase {
public:
 explicit MPIContext(MPI_Comm comm, const std::string& cfg)
     : ContextBase(cfg, comm_rank(comm)), m_comm(comm) {}

 MPIContext(const MPIContext &)     = delete;
 MPIContext(MPIContext &&) noexcept = default;

 MPIContext &operator=(const MPIContext &) = delete;
 MPIContext &operator=(MPIContext &&) noexcept = default;

 virtual ~MPIContext() {
#ifdef KR_ENABLE_TRACING
    int rank = -1;
    MPI_Comm_rank(m_comm, &rank);
    int size = -1;
    MPI_Comm_size(m_comm, &size);

    std::ostringstream fname;
    fname << "trace" << rank << ".json";

    std::ofstream out(fname.str());

    std::cout << "writing trace to " << fname.str() << '\n';

    trace().write(out);

    // Metafile
    picojson::object root;
    root["num_ranks"] = picojson::value(static_cast<double>(size));

    std::ofstream meta_out("meta.json");
    picojson::value(root).serialize(std::ostream_iterator<char>(meta_out),
                                    true);
#endif
 }

  MPI_Comm comm() const noexcept { return m_comm; }

  //TODO: Allow configuring global vs local consistency requirements

  bool restart_available(const std::string &label, int version) override {
    int avail, locally_avail = m_backend->restart_available(label, version);
    MPI_Allreduce(&locally_avail, &avail, 1, MPI_INT, MPI_LAND, m_comm);

    return avail;
  }

  void restart(const std::string &label, int version,
               const std::unordered_set<Registration> &members) override {
    int success = m_backend->restart(label, version, members);
    MPI_Allreduce(MPI_IN_PLACE, &success, 1, MPI_INT, MPI_LAND, m_comm);

    //TODO: configurability on this?
    //  Throwing and using an exception handler seems like it'd
    //  make for awkwardly-wrapped user code
    if(!success) MPI_Abort(m_comm, 1);
  }

  void checkpoint(const std::string &label, int version,
                  const std::unordered_set<KokkosResilience::Registration>
                      &members) override {
    //Ignore success status, if we failed we just hope to run long enough to
    //succeed next time.
    m_backend->checkpoint(label, version, members);
  }

  int latest_version(const std::string &label) const noexcept override {
    int latest = m_backend->latest_version(label);
    MPI_Allreduce(MPI_IN_PLACE, &latest, 1, MPI_INT, MPI_MIN, m_comm);
    return latest;
  }

  void reset() override { m_backend->reset(); }

private:
  MPI_Comm m_comm;
};

} // namespace KokkosResilience

#endif // INC_KOKKOS_RESILIENCE_MPICONTEXT_HPP
