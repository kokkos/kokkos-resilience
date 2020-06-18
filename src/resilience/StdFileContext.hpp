#ifndef INC_KOKKOS_RESILIENCE_STDFILECONTEXT_HPP
#define INC_KOKKOS_RESILIENCE_STDFILECONTEXT_HPP

#include "Context.hpp"

#include <string>

namespace KokkosResilience {

template <typename Backend>
class StdFileContext : public ContextBase {
 public:
  explicit StdFileContext(std::string const &filename, Config &cfg)
      : ContextBase(cfg), m_backend(*this, filename), m_filename(filename) {}

  StdFileContext(const StdFileContext &) = delete;
  StdFileContext(StdFileContext &&)      = default;

  StdFileContext &operator=(const StdFileContext &) = delete;
  StdFileContext &operator=(StdFileContext &&) = default;

  virtual ~StdFileContext() {
#ifdef KR_ENABLE_TRACING
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

  std::string const &filename() const noexcept { return m_filename; }

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

  void reset() override {
    m_backend.reset();
  }

 private:
  std::string m_filename;
  Backend m_backend;
};

}  // namespace KokkosResilience

#endif  // INC_KOKKOS_RESILIENCE_STDFILECONTEXT_HPP
