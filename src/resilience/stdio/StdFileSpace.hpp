#ifndef INC_RESILIENCE_STDIO_STDFILESPACE_HPP
#define INC_RESILIENCE_STDIO_STDFILESPACE_HPP

#include <cstring>
#include <string>
#include <iosfwd>
#include <typeinfo>

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Concepts.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>
#include "resilience/filesystem/ExternalIOInterface.hpp"
#include <fstream>


namespace KokkosResilience {

class KokkosStdFileAccessor : public KokkosIOAccessor {


public:
   size_t file_offset;
   std::fstream file_strm;


   KokkosStdFileAccessor() : KokkosIOAccessor(),
                             file_offset(0) {
   }
   KokkosStdFileAccessor(const size_t size, const std::string & path ) : KokkosIOAccessor(size, path, true),
                                                                         file_offset(0) {
   }

   KokkosStdFileAccessor( const KokkosStdFileAccessor & rhs ) = default;
   KokkosStdFileAccessor( KokkosStdFileAccessor && rhs ) = default;
   KokkosStdFileAccessor & operator = ( KokkosStdFileAccessor && ) = default;
   KokkosStdFileAccessor & operator = ( const KokkosStdFileAccessor & ) = default;
   KokkosStdFileAccessor( void* ptr ) {
      KokkosStdFileAccessor * pAcc = static_cast<KokkosStdFileAccessor*>(ptr);
      if (pAcc) {
         data_size = pAcc->data_size;
         file_path = pAcc->file_path;
         file_offset = pAcc->file_offset;
      }

   }

   KokkosStdFileAccessor( void* ptr, const size_t offset ) {
      KokkosStdFileAccessor * pAcc = static_cast<KokkosStdFileAccessor*>(ptr);
      if (pAcc) {
         data_size = pAcc->data_size;
         file_path = pAcc->file_path;
         file_offset = offset;
      }
   }

   int initialize( const std::string & filepath );

   bool open_file(int read_write = KokkosStdFileAccessor::READ_FILE);
   void close_file();

   virtual size_t ReadFile_impl(void * dest, const size_t dest_size);
   
   virtual size_t WriteFile_impl(const void * src, const size_t src_size);
   
   virtual size_t OpenFile_impl();

   void finalize();
   
   virtual ~KokkosStdFileAccessor() {
   }
};


/// \class StdFileSpace
/// \brief Memory management for StdFile
///
/// StdFileSpace is a memory space that governs access to StdFile data.
///
class StdFileSpace {
public:
  //! Tag this class as a kokkos memory space
  typedef KokkosResilience::StdFileSpace  file_space;   // used to uniquely identify file spaces
  typedef KokkosResilience::StdFileSpace  memory_space;
  typedef size_t     size_type;

  /// \typedef execution_space
  /// \brief Default execution space for this memory space.
  ///
  /// Every memory space has a default execution space.  This is
  /// useful for things like initializing a View (which happens in
  /// parallel using the View's default execution space).
#if defined( KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_OPENMP )
  typedef Kokkos::OpenMP    execution_space;
#elif defined( KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_THREADS )
  typedef Kokkos::Threads   execution_space;
//#elif defined( KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_QTHREADS )
//  typedef Kokkos::Qthreads  execution_space;
#elif defined( KOKKOS_ENABLE_OPENMP )
  typedef Kokkos::OpenMP    execution_space;
#elif defined( KOKKOS_ENABLE_THREADS )
  typedef Kokkos::Threads   execution_space;
//#elif defined( KOKKOS_ENABLE_QTHREADS )
//  typedef Kokkos::Qthreads  execution_space;
#elif defined( KOKKOS_ENABLE_SERIAL )
  typedef Kokkos::Serial    execution_space;
#else
#  error "At least one of the following host execution spaces must be defined: Kokkos::OpenMP, Kokkos::Threads, Kokkos::Qthreads, or Kokkos::Serial.  You might be seeing this message if you disabled the Kokkos::Serial device explicitly using the Kokkos_ENABLE_Serial:BOOL=OFF CMake option, but did not enable any of the other host execution space devices."
#endif

  //! This memory space preferred device_type
  typedef Kokkos::Device< execution_space, memory_space > device_type;

  /**\brief  Default memory space instance */
  StdFileSpace();
  StdFileSpace( StdFileSpace && rhs ) = default;
  StdFileSpace( const StdFileSpace & rhs ) = default;
  StdFileSpace & operator = ( StdFileSpace && ) = default;
  StdFileSpace & operator = ( const StdFileSpace & ) = default;
  ~StdFileSpace() = default;

  /**\brief  Allocate untracked memory in the space */
  void * allocate( const size_t arg_alloc_size, const std::string & path ) const;

  /**\brief  Deallocate untracked memory in the space */
  void deallocate( void * const arg_alloc_ptr
                 , const size_t arg_alloc_size ) const;

  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return m_name; }

  static void restore_all_views();
  static void restore_view(const std::string name);
  static void checkpoint_views();
  static void checkpoint_create_view_targets();
  static void set_default_path( const std::string path );
  static std::string s_default_path;

private:
  static constexpr const char* m_name = "StdFile";
  friend class Kokkos::Impl::SharedAllocationRecord< KokkosResilience::StdFileSpace, void >;
};

} // namespace KokkosResilience

namespace Kokkos {

namespace Impl {

template<>
class SharedAllocationRecord< KokkosResilience::StdFileSpace, void >
  : public SharedAllocationRecord< void, void >
{
private:
  friend KokkosResilience::StdFileSpace;

  typedef SharedAllocationRecord< void, void >  RecordBase;

  SharedAllocationRecord( const SharedAllocationRecord & ) = delete;
  SharedAllocationRecord & operator = ( const SharedAllocationRecord & ) = delete;

  static void deallocate( RecordBase * );

#ifdef KOKKOS_DEBUG
  /**\brief  Root record for tracked allocations from this StdFileSpace instance */
  static RecordBase s_root_record;
#endif

  const KokkosResilience::StdFileSpace m_space;

protected:
  ~SharedAllocationRecord();
  SharedAllocationRecord() = default;

  SharedAllocationRecord( const KokkosResilience::StdFileSpace        & arg_space
                        , const std::string              & arg_label
                        , const size_t                     arg_alloc_size
                        , const RecordBase::function_type  arg_dealloc = & deallocate
                        );

public:

  inline
  std::string get_label() const
  {
    return std::string( RecordBase::head()->m_label );
  }

  KOKKOS_INLINE_FUNCTION static
  SharedAllocationRecord * allocate( const KokkosResilience::StdFileSpace &  arg_space
                                   , const std::string       &  arg_label
                                   , const size_t               arg_alloc_size
                                   )
  {
#if defined( KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST )
    return new SharedAllocationRecord( arg_space, arg_label, arg_alloc_size );
#else
    return (SharedAllocationRecord *) 0;
#endif
  }


  /**\brief  Allocate tracked memory in the space */
  static
  void * allocate_tracked( const KokkosResilience::StdFileSpace & arg_space
                         , const std::string & arg_label
                         , const size_t arg_alloc_size );

  /**\brief  Reallocate tracked memory in the space */
  static
  void * reallocate_tracked( void * const arg_alloc_ptr
                           , const size_t arg_alloc_size );

  /**\brief  Deallocate tracked memory in the space */
  static
  void deallocate_tracked( void * const arg_alloc_ptr );

  static SharedAllocationRecord * get_record( void * arg_alloc_ptr );

  static void print_records( std::ostream &, const KokkosResilience::StdFileSpace &, bool detail = false );
};


template<class ExecutionSpace> struct DeepCopy< KokkosResilience::StdFileSpace , Kokkos::HostSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  {
    KokkosResilience::KokkosIOAccessor::transfer_from_host( dst, src, n );
  }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    KokkosResilience::KokkosIOAccessor::transfer_from_host( dst, src, n );
  }
};

template<class ExecutionSpace> struct DeepCopy<  Kokkos::HostSpace , KokkosResilience::StdFileSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  {
    KokkosResilience::KokkosIOAccessor::transfer_to_host( dst, src, n );
  }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    KokkosResilience::KokkosIOAccessor::transfer_to_host( dst, src, n );
  }
};

} // Impl

} // Kokkos

#include "resilience/filesystem/DirectoryManagement.hpp"
#endif  // INC_RESILIENCE_STDIO_STDFILESPACE_HPP
