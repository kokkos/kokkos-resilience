#include "Filesystem.hpp"

#include <boost/filesystem.hpp>

namespace KokkosResilience
{
  std::uintmax_t remove_all( const std::string &_path )
  {
    return boost::filesystem::remove_all( _path );
  }
  
  bool create_directory( const std::string &_path )
  {
    return boost::filesystem::create_directory( _path );
  }

  bool file_exists( const std::string &_path )
  {
    return boost::filesystem::exists( _path );
  }
}
