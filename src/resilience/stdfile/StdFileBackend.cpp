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

std::string versionless_filename(std::string const &filename, std::string const &label) {
  return filename + "." + label;
}

std::string full_filename(std::string const &filename, std::string const &label,
                          int version) {
  return versionless_filename(filename, label) + "." + std::to_string(version);
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
  std::string filename = detail::versionless_filename(m_filename, label);
  boost::filesystem::path dir(filename);

  filename = dir.filename().string();

  dir = boost::filesystem::absolute(dir).parent_path();

  for(auto &entry : boost::filesystem::directory_iterator(dir)){
    if (!boost::filesystem::is_regular_file(entry)) {
      continue;
    }
    if(filename == entry.path().filename().stem().string()){
        //This is a checkpoint, probably.
        try{
            int vers = std::stoi(entry.path().filename().extension().string().substr(1));
            result = std::max(result,vers);
        } catch(...) {
            //Just not the filename format we expected, could be unrelated.
        }
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
