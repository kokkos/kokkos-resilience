

#ifndef INC_RESILIENCE_FILESSYSTEM_EXTERNALIOINTERFACE_HPP
#define INC_RESILIENCE_FILESSYSTEM_EXTERNALIOINTERFACE_HPP

#include <cstring>
#include <string>
#include <iosfwd>
#include <typeinfo>

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Concepts.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <map>

namespace KokkosResilience {

class KokkosIOAccessor  {

public:
   enum { READ_FILE = 0,
          WRITE_FILE = 1 };

   size_t data_size;
   bool is_contiguous;
   std::string file_path;

   KokkosIOAccessor() : data_size(0),
                        is_contiguous(true),
                        file_path("") {
   }
   KokkosIOAccessor(const size_t size, const std::string & path, bool cont_ = true ) : data_size(size),
                                                                    is_contiguous(cont_),
                                                                    file_path(path) {
   }

   KokkosIOAccessor( const KokkosIOAccessor & rhs ) = default;
   KokkosIOAccessor( KokkosIOAccessor && rhs ) = default;
   KokkosIOAccessor & operator = ( KokkosIOAccessor && ) = default;
   KokkosIOAccessor & operator = ( const KokkosIOAccessor & ) = default;

   size_t ReadFile(void * dest, const size_t dest_size) {
      return ReadFile_impl( dest, dest_size );
   }
   
   size_t WriteFile(const void * src, const size_t src_size) {
      return WriteFile_impl( src, src_size );
   }

   size_t OpenFile() {
      return OpenFile_impl();
   }

   virtual size_t ReadFile_impl(void * dest, const size_t dest_size) = 0;
   
   virtual size_t WriteFile_impl(const void * src, const size_t src_size) = 0;

   virtual size_t OpenFile_impl() = 0;
   
   virtual ~KokkosIOAccessor() {
   }

   static std::string resolve_path( std::string path, std::string default_ );
   static void transfer_from_host ( void * dst, const void * src, size_t size_ );
   static void transfer_to_host ( void * dst, const void * src, size_t size_ );
   static void create_empty_file ( void * dst );
};

struct KokkosIOInterface : Kokkos::Impl::SharedAllocationHeader {
   KokkosIOAccessor * pAcc;
};


class KokkosIOConfigurationManager {
public:
   std::map<std::string, boost::property_tree::ptree> m_config_list;
   void load_configuration ( std::string path );
   boost::property_tree::ptree get_config( std::string name ) {
      return m_config_list[name];
   }

   static KokkosIOConfigurationManager * get_instance();

   static KokkosIOConfigurationManager * m_Inst;

};

} // namespace KokkosResilience


#endif // INC_RESILIENCE_FILESSYSTEM_EXTERNALIOINTERFACE_HPP
