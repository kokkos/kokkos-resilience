#ifndef INC_KOKKOS_RESILIENCE_MPICONTEXT_HPP
#define INC_KOKKOS_RESILIENCE_MPICONTEXT_HPP

#include <mpi.h>
#include "Context.hpp"

namespace KokkosResilience {

template <typename Backend>
class MPIContext : public ContextBase {
public:
  explicit MPIContext(MPI_Comm comm, Config &cfg)
      : ContextBase(cfg), m_backend(*this, comm), m_comm(comm) {}

  MPIContext(const MPIContext &) = delete;
  MPIContext(MPIContext &&) = default;

  MPIContext &operator=(const MPIContext &) = delete;
  MPIContext &operator=(MPIContext &&) = default;

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
      const std::vector<std::unique_ptr<Kokkos::ViewHolderBase>> &views,
      const std::vector<Detail::CrefImpl> &crefs) override {
    m_backend.register_hashes(views, crefs);
  }

  bool restart_available(const std::string &label, int version) override {
    return m_backend.restart_available(label, version);
  }

  void restart(const std::string &label, int version,
               const std::vector<std::unique_ptr<Kokkos::ViewHolderBase>>
                   &views) override {
    m_backend.restart(label, version, views);
  }

  void checkpoint(const std::string &label, int version,
                  const std::vector<std::unique_ptr<Kokkos::ViewHolderBase>>
                      &views) override {
    m_backend.checkpoint(label, version, views);
  }

  int latest_version(const std::string &label) const noexcept override {
    return m_backend.latest_version(label);
  }

  void reset() override { m_backend.reset(); }

private:
  MPI_Comm m_comm;
  Backend m_backend;
};

} // namespace KokkosResilience

#endif // INC_KOKKOS_RESILIENCE_MPICONTEXT_HPP
