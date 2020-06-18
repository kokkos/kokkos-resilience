#ifndef INC_RESILIENCE_STDFILE_STDFILEBACKEND_HPP
#define INC_RESILIENCE_STDFILE_STDFILEBACKEND_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_ViewHooks.hpp>

#include <memory>
#include <string>
#include <vector>

#include "../Cref.hpp"

namespace KokkosResilience {
template <typename Backend>
class StdFileContext;

class StdFileBackend {
 public:
  StdFileBackend(StdFileContext<StdFileBackend> &ctx,
                 std::string const &filename);
  ~StdFileBackend();

  void checkpoint(
      const std::string &label, int version,
      const std::vector<std::unique_ptr<Kokkos::ViewHolderBase>> &views);

  bool restart_available(const std::string &label, int version);
  int latest_version(const std::string &label) const noexcept;

  void restart(
      const std::string &label, int version,
      const std::vector<std::unique_ptr<Kokkos::ViewHolderBase>> &views);

  void reset() {}

  void register_hashes(
      const std::vector<std::unique_ptr<Kokkos::ViewHolderBase>> &views,
      const std::vector<Detail::CrefImpl> &crefs) {}

 private:
  std::string m_filename;
  StdFileContext<StdFileBackend> &m_context;
};

}  // namespace KokkosResilience

#endif  // INC_RESILIENCE_STDFILE_STDFILEBACKEND_HPP
