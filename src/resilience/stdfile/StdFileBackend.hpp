#ifndef INC_RESILIENCE_STDFILE_STDFILEBACKEND_HPP
#define INC_RESILIENCE_STDFILE_STDFILEBACKEND_HPP

#include <Kokkos_Core.hpp>
#include "../view_hooks/ViewHolder.hpp"

#include <memory>
#include <string>
#include <vector>

#include "../Cref.hpp"
#include "../StdFileContext.hpp"

namespace KokkosResilience {

class StdFileBackend {
 public:
  StdFileBackend(StdFileContext<StdFileBackend> &ctx,
                 std::string const &filename);
  ~StdFileBackend();

  void checkpoint(
      const std::string &label, int version,
      const std::vector< KokkosResilience::ViewHolder > &views);

  bool restart_available(const std::string &label, int version);
  int latest_version(const std::string &label) const noexcept;

  void restart(
      const std::string &label, int version,
      const std::vector< KokkosResilience::ViewHolder > &views);

  void reset() {}

  void register_hashes(
      const std::vector< KokkosResilience::ViewHolder > &views,
      const std::vector<Detail::CrefImpl> &crefs) {}

 private:
  std::string m_filename;
  StdFileContext<StdFileBackend> &m_context;
};

}  // namespace KokkosResilience

#endif  // INC_RESILIENCE_STDFILE_STDFILEBACKEND_HPP
