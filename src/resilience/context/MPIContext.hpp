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
#include "Context.hpp"

namespace KokkosResilience {

template <typename Backend>
class MPIContext : public ContextBase {
public:
 explicit MPIContext(MPI_Comm comm, const Config &cfg)
     : ContextBase(cfg), m_comm(comm), m_backend(*this, comm) {}

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

  Backend &backend() { return m_backend; }

  void register_hashes(
      const std::vector<KokkosResilience::ViewHolder> &views,
      const std::vector<Detail::CrefImpl> &crefs) override {
    m_backend.register_hashes(views, crefs);
  }

  bool restart_available(const std::string &label, int version) override {
    return m_backend.restart_available(label, version);
  }

  void restart(const std::string &label, int version,
               const std::vector<KokkosResilience::ViewHolder>
                   &views) override {
    m_backend.restart(label, version, views);
  }

  void checkpoint(const std::string &label, int version,
                  const std::vector<KokkosResilience::ViewHolder>
                      &views) override {
    m_backend.checkpoint(label, version, views);
  }

  int latest_version(const std::string &label) const noexcept override {
    return m_backend.latest_version(label);
  }

  void register_alias( const std::string &original, const std::string &alias ) override {
    return m_backend.register_alias( original, alias );
  }

  void reset() override { m_backend.reset(); }

private:
  MPI_Comm m_comm;
  Backend m_backend;
};

} // namespace KokkosResilience

#endif // INC_KOKKOS_RESILIENCE_MPICONTEXT_HPP
