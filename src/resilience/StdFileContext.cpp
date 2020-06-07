#include "StdFileContext.hpp"
#include "stdfile/StdFileBackend.hpp"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace KokkosResilience {
std::unique_ptr<ContextBase> make_context(const std::string& filename,
                                          const std::string& config) {
  auto cfg = Config{config};

  using fun_type = std::function<std::unique_ptr<ContextBase>()>;
  static std::unordered_map<std::string, fun_type> backends = {
      {"stdfile", [&]() {
         return std::make_unique<StdFileContext<StdFileBackend> >(filename, cfg);
       }}};

  auto pos = backends.find(cfg["backend"].as<std::string>());
  if (pos == backends.end()) return std::unique_ptr<ContextBase>{};

  return pos->second();
}
}  // namespace KokkosResilience
