#ifndef INC_RESILIENCE_CUDA_RESCUDASPACE_HPP
#define INC_RESILIENCE_CUDA_RESCUDASPACE_HPP

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ENABLE_CUDA )

#include <impl/Kokkos_TrackDuplicates.hpp>
#include <Kokkos_CudaSpace.hpp>
#include <cmath>
#include <map>
#include <typeinfo>

/*--------------------------------------------------------------------------*/

namespace KokkosResilience {


/** \brief  cuda on-device memory management */

class ResCudaSpace : public Kokkos::CudaSpace {
public:
  //! Tag this class as a kokkos memory space
  typedef ResCudaSpace             memory_space ;
  typedef ResCudaSpace          resilient_space ;
  typedef Kokkos::Cuda          execution_space ;
  typedef Kokkos::Device<execution_space,memory_space> device_type;

  typedef unsigned int          size_type ;

  /*--------------------------------*/

  ResCudaSpace();
  ResCudaSpace( ResCudaSpace && rhs ) = default ;
  ResCudaSpace( const ResCudaSpace & rhs ) = default ;
  ResCudaSpace & operator = ( ResCudaSpace && rhs ) = default ;
  ResCudaSpace & operator = ( const ResCudaSpace & rhs ) = default ;
  ~ResCudaSpace() = default ;
  
  static void clear_duplicates_list();

  static std::map<std::string, Kokkos::Experimental::DuplicateTracker * > duplicate_map;

private:

  friend class Kokkos::Impl::SharedAllocationRecord< ResCudaSpace , void > ;
};

} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

static_assert( Kokkos::Impl::MemorySpaceAccess< KokkosResilience::ResCudaSpace , KokkosResilience::ResCudaSpace >::assignable , "" );

//----------------------------------------

template<>
struct MemorySpaceAccess< Kokkos::HostSpace , KokkosResilience::ResCudaSpace > {
  enum { assignable = false };
  enum { accessible = false };
  enum { deepcopy   = true };
};

//----------------------------------------

template<>
struct MemorySpaceAccess< KokkosResilience::ResCudaSpace , Kokkos::HostSpace > {
  enum { assignable = false };
  enum { accessible = false };
  enum { deepcopy   = true };
};

template<>
struct MemorySpaceAccess< Kokkos::CudaSpace , KokkosResilience::ResCudaSpace > {
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy   = true };
};

//----------------------------------------

template<>
struct MemorySpaceAccess< KokkosResilience::ResCudaSpace , Kokkos::CudaSpace > {
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy   = true };
};

template<>
struct MemorySpaceAccess< KokkosResilience::ResCudaSpace , Kokkos::CudaUVMSpace > {
  // CudaSpace::execution_space == CudaUVMSpace::execution_space
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy   = true };
};

template<>
struct MemorySpaceAccess< KokkosResilience::ResCudaSpace , Kokkos::CudaHostPinnedSpace > {
  // CudaSpace::execution_space != CudaHostPinnedSpace::execution_space
  enum { assignable = false };
  enum { accessible = true }; // ResCudaSpace::execution_space
  enum { deepcopy   = true };
};

//----------------------------------------
// CudaUVMSpace::execution_space == cuda
// CudaUVMSpace accessible to both cuda and Host

template<>
struct MemorySpaceAccess< Kokkos::CudaUVMSpace , KokkosResilience::ResCudaSpace > {
  // CudaUVMSpace::execution_space == CudaSpace::execution_space
  // Can access CudaUVMSpace from Host but cannot access ResCudaSpace from Host
  enum { assignable = false };

  // CudaUVMSpace::execution_space can access CudaSpace
  enum { accessible = true };
  enum { deepcopy   = true };
};

template<>
struct MemorySpaceAccess< Kokkos::CudaHostPinnedSpace , KokkosResilience::ResCudaSpace > {
  enum { assignable = false }; // Cannot access from Host
  enum { accessible = false };
  enum { deepcopy   = true };
};

//----------------------------------------

}} // namespace Kokkos::Impl

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

template<> struct DeepCopy< KokkosResilience::ResCudaSpace , KokkosResilience::ResCudaSpace , Cuda>
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Cuda & , void * dst , const void * src , size_t );
};

template<> struct DeepCopy< KokkosResilience::ResCudaSpace , HostSpace , Cuda >
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Cuda & , void * dst , const void * src , size_t );
};

template<> struct DeepCopy< HostSpace , KokkosResilience::ResCudaSpace , Cuda >
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Cuda & , void * dst , const void * src , size_t );
};

template<class ExecutionSpace> struct DeepCopy< KokkosResilience::ResCudaSpace , KokkosResilience::ResCudaSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< KokkosResilience::ResCudaSpace , KokkosResilience::ResCudaSpace , Cuda >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncCuda (dst,src,n);
  }
};

template<class ExecutionSpace> struct DeepCopy< KokkosResilience::ResCudaSpace , HostSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< KokkosResilience::ResCudaSpace , HostSpace , Cuda>( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncCuda (dst,src,n);
  }
};

template<class ExecutionSpace>
struct DeepCopy< HostSpace , KokkosResilience::ResCudaSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< HostSpace , KokkosResilience::ResCudaSpace , Cuda >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncCuda (dst,src,n);
  }
};

template<class ExecutionSpace>
struct DeepCopy< KokkosResilience::ResCudaSpace , CudaSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< CudaSpace , CudaSpace , Cuda >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncCuda (dst,src,n);
  }
};

template<class ExecutionSpace>
struct DeepCopy< CudaSpace , KokkosResilience::ResCudaSpace , ExecutionSpace>
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< CudaSpace , CudaSpace , Cuda >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncCuda (dst,src,n);
  }
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

/** Running in ResCudaSpace attempting to access HostSpace: error */
template<>
struct VerifyExecutionCanAccessMemorySpace< KokkosResilience::ResCudaSpace , Kokkos::HostSpace >
{
  enum { value = false };
  KOKKOS_INLINE_FUNCTION static void verify( void )
    { Kokkos::abort("cuda code attempted to access HostSpace memory"); }

  KOKKOS_INLINE_FUNCTION static void verify( const void * )
    { Kokkos::abort("cuda code attempted to access HostSpace memory"); }
};

/** Running in ResCudaSpace accessing CudaUVMSpace: ok */
template<>
struct VerifyExecutionCanAccessMemorySpace< KokkosResilience::ResCudaSpace , Kokkos::CudaUVMSpace >
{
  enum { value = true };
  KOKKOS_INLINE_FUNCTION static void verify( void ) { }
  KOKKOS_INLINE_FUNCTION static void verify( const void * ) { }
};

/** Running in CudaSpace accessing CudaHostPinnedSpace: ok */
template<>
struct VerifyExecutionCanAccessMemorySpace< KokkosResilience::ResCudaSpace , Kokkos::CudaHostPinnedSpace >
{
  enum { value = true };
  KOKKOS_INLINE_FUNCTION static void verify( void ) { }
  KOKKOS_INLINE_FUNCTION static void verify( const void * ) { }
};

/** Running in CudaSpace attempting to access an unknown space: error */
template< class OtherSpace >
struct VerifyExecutionCanAccessMemorySpace<
  typename enable_if< ! is_same<KokkosResilience::ResCudaSpace,OtherSpace>::value , KokkosResilience::ResCudaSpace >::type ,
  OtherSpace >
{
  enum { value = false };
  KOKKOS_INLINE_FUNCTION static void verify( void )
    { Kokkos::abort("cuda code attempted to access unknown Space memory"); }

  KOKKOS_INLINE_FUNCTION static void verify( const void * )
    { Kokkos::abort("cuda code attempted to access unknown Space memory"); }
};

//----------------------------------------------------------------------------
/** Running in HostSpace attempting to access CudaSpace */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::HostSpace , KokkosResilience::ResCudaSpace >
{
  enum { value = false };
  inline static void verify( void ) { KokkosResilience::ResCudaSpace::access_error(); }
  inline static void verify( const void * p ) { KokkosResilience::ResCudaSpace::access_error(p); }
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template<>
class SharedAllocationRecord< KokkosResilience::ResCudaSpace , void >
  : public SharedAllocationRecord< void , void >
{
private:

  friend class SharedAllocationRecord< Kokkos::CudaUVMSpace , void > ;

  typedef SharedAllocationRecord< void , void >  RecordBase ;

  SharedAllocationRecord( const SharedAllocationRecord & ) = delete ;
  SharedAllocationRecord & operator = ( const SharedAllocationRecord & ) = delete ;

  static void deallocate( RecordBase * );

  static ::cudaTextureObject_t
  attach_texture_object( const unsigned sizeof_alias
                       , void * const   alloc_ptr
                       , const size_t   alloc_size );

#ifdef KOKKOS_DEBUG
  static RecordBase s_root_record ;
#endif

  ::cudaTextureObject_t   m_tex_obj ;
  const KokkosResilience::ResCudaSpace m_space ;

protected:

  ~SharedAllocationRecord();
  SharedAllocationRecord() : RecordBase(), m_tex_obj(0), m_space() {}

  SharedAllocationRecord( const KokkosResilience::ResCudaSpace        & arg_space
                        , const std::string              & arg_label
                        , const size_t                     arg_alloc_size
                        , const RecordBase::function_type  arg_dealloc = & deallocate
                        );

public:
  KokkosResilience::ResCudaSpace get_space() const { return m_space; }
  std::string get_label() const ;

  static SharedAllocationRecord * allocate( const KokkosResilience::ResCudaSpace &  arg_space
                                          , const std::string       &  arg_label
                                          , const size_t               arg_alloc_size );

  /**\brief  Allocate tracked memory in the space */
  static
  void * allocate_tracked( const KokkosResilience::ResCudaSpace & arg_space
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

  template< typename AliasType >
  inline
  ::cudaTextureObject_t attach_texture_object()
    {
      static_assert( ( std::is_same< AliasType , int >::value ||
                       std::is_same< AliasType , ::int2 >::value ||
                       std::is_same< AliasType , ::int4 >::value )
                   , "cuda texture fetch only supported for alias types of int, ::int2, or ::int4" );

      if ( m_tex_obj == 0 ) {
        m_tex_obj = attach_texture_object( sizeof(AliasType)
                                         , (void*) RecordBase::m_alloc_ptr
                                         , RecordBase::m_alloc_size );
      }

      return m_tex_obj ;
    }

  template< typename AliasType >
  inline
  int attach_texture_object_offset( const AliasType * const ptr )
    {
      // Texture object is attached to the entire allocation range
      return ptr - reinterpret_cast<AliasType*>( RecordBase::m_alloc_ptr );
    }

  static void print_records( std::ostream & , const KokkosResilience::ResCudaSpace & , bool detail = false );
};



} // namespace Impl
} // namespace Kokkos

namespace KokkosResilience {
//  template <class DType, class ExecSpace> void * CombineFunctor<DType,ExecSpace>::s_dup_kernel = nullptr;

  template<class Functor>
  __global__ static void launch_comb_dup_kernel( const Functor cf ) {
       int iwork = threadIdx.y + blockDim.y * blockIdx.x ;
       if ( iwork <  cf.get_len() )
          cf.exec(iwork);
  }
 
} // namespace KokkosResilience


#define KR_MAKE_RESILIENCE_FUNC_NAME( id ) id##_resilience_func

#define KR_DECLARE_RESILIENCE_OBJECTS(data_type, id) \
   template __global__ void KokkosResilience::launch_comb_dup_kernel<Kokkos::Experimental::CombineFunctor<data_type, KokkosResilience::ResCuda> >( \
                                                                Kokkos::Experimental::CombineFunctor<data_type, KokkosResilience::ResCuda> ); \
   void * KR_MAKE_RESILIENCE_FUNC_NAME(id) = (void*)&KokkosResilience::launch_comb_dup_kernel<Kokkos::Experimental::CombineFunctor<data_type, KokkosResilience::ResCuda> >;

#define KR_ADD_RESILIENCE_OBJECTS(data_type, id) \
   Kokkos::Experimental::DuplicateTracker::add_kernel_func( typeid(data_type).name(), KR_MAKE_RESILIENCE_FUNC_NAME( id ));


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_ENABLE_CUDA ) */
#endif// INC_RESILIENCE_CUDA_RESCUDASPACE_HPP

