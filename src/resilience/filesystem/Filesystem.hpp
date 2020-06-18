#ifndef INC_RESILIENCE_FILESYSTEM_FILESYSTEM_HPP
#define INC_RESILIENCE_FILESYSTEM_FILESYSTEM_HPP

#include <string>
#include <cstdint>

namespace KokkosResilience
{
  std::uintmax_t remove_all( const std::string &_path );
  bool create_directory( const std::string &_path );
  bool file_exists( const std::string &_path );
}

#endif  // INC_RESILIENCE_FILESYSTEM_FILESYSTEM_HPP
