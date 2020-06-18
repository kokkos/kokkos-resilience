#include "StdFileBackend.hpp"

#include <cassert>
#include <fstream>
#include <string>

#include <boost/filesystem.hpp>

#include "../AutomaticCheckpoint.hpp"

#ifdef KR_ENABLE_TRACING
#include "../util/Trace.hpp"
#endif

namespace KokkosResilience {

namespace detail {

std::string full_filename(std::string const &filename, std::string const &label,
                          int version) {
  return filename + "." + label + "." + std::to_string(version);
}
}  // namespace detail

StdFileBackend::StdFileBackend(StdFileContext<StdFileBackend> &ctx,
                               std::string const &filename)
    : m_context(ctx), m_filename(filename) {}

StdFileBackend::~StdFileBackend() = default;

void StdFileBackend::checkpoint(
    const std::string &label, int version,
    const std::vector<std::unique_ptr<Kokkos::ViewHolderBase>> &views) {
  bool status = true;
  try {
    std::string filename = detail::full_filename(m_filename, label, version);
    std::ofstream file(filename, std::ios::binary);

#ifdef KR_ENABLE_TRACING
    auto write_trace =
        Util::begin_trace<Util::TimingTrace<std::string>>(m_context, "write");
#endif
    for (auto &&v : views) {
      char *bytes     = static_cast<char *>(v->data());
      std::size_t len = v->span() * v->data_type_size();

      file.write(bytes, len);
    }
#ifdef KR_ENABLE_TRACING
    write_trace.end();
#endif
  } catch (...) {
    status = false;
  }
}

bool StdFileBackend::restart_available(const std::string &label, int version) {
  std::string filename = detail::full_filename(m_filename, label, version);
  return boost::filesystem::exists(filename);
}

int StdFileBackend::latest_version(const std::string &label) const noexcept {
  int result = -1;
  for (int version = 0; /**/; ++version) {
    std::string filename = detail::full_filename(m_filename, label, version);
    if (!boost::filesystem::exists(filename)) {
      result = version - 1;
      break;
    }
  }
  return result;
}

void StdFileBackend::restart(
    const std::string &label, int version,
    const std::vector<std::unique_ptr<Kokkos::ViewHolderBase>> &views) {
  bool status = true;
  try {
    std::string filename = detail::full_filename(m_filename, label, version);
    std::ifstream file(filename, std::ios::binary);

#ifdef KR_ENABLE_TRACING
    auto read_trace =
        Util::begin_trace<Util::TimingTrace<std::string>>(m_context, "read");
#endif
    for (auto &&v : views) {
      char *bytes     = static_cast<char *>(v->data());
      std::size_t len = v->span() * v->data_type_size();

      file.read(bytes, len);
    }
#ifdef KR_ENABLE_TRACING
    read_trace.end();
#endif
  } catch (...) {
    status = false;
  }
}
}  // namespace KokkosResilience
