#ifndef INC_RESILIENCE_VELOC_VELOCBACKEND_HPP
#define INC_RESILIENCE_VELOC_VELOCBACKEND_HPP

#include <string>
#include <vector>
#include <memory>
#include <Kokkos_Core.hpp>
#include <Kokkos_ViewHooks.hpp>
#include <unordered_map>
#include <unordered_set>
#include <mpi.h>
#include "../Cref.hpp"

namespace KokkosResilience
{
  template< typename Backend >
  class MPIContext;

  namespace Detail
  {
    struct MemProtectKey
    {
      explicit MemProtectKey( void *maddr )
          : addr( reinterpret_cast< std::uintptr_t >( maddr ) )
      {}

      std::uintptr_t addr;

      friend bool operator==( const MemProtectKey &_lhs, const MemProtectKey &_rhs )
      {
        return _lhs.addr == _rhs.addr;
      }

      friend bool operator!=( const MemProtectKey &_lhs, const MemProtectKey &_rhs )
      {
        return !( _lhs == _rhs );
      }

      friend bool operator<( const MemProtectKey &_lhs, const MemProtectKey &_rhs )
      {
        return _lhs.addr < _rhs.addr;
      }
    };

    struct MemProtectBlock
    {
      explicit MemProtectBlock( int mid )
          : id( mid )
      {}

      explicit MemProtectBlock( int mid, std::vector< unsigned char > &&mbuff )
          : id( mid ), buff( std::move( mbuff ) )
      {}

      int id;
      std::vector< unsigned char > buff;
    };
  }
}



namespace std
{
  template<>
  struct hash< KokkosResilience::Detail::MemProtectKey >
  {
    std::size_t operator()( const KokkosResilience::Detail::MemProtectKey &_mem ) const noexcept
    {
      return std::hash< std::uintptr_t >{}( _mem.addr );
    }
  };
}

namespace KokkosResilience
{
  class VeloCMemoryBackend
  {
  public:
    
    using context_type = MPIContext< VeloCMemoryBackend >;
    
    VeloCMemoryBackend( context_type &ctx, MPI_Comm mpi_comm );
    ~VeloCMemoryBackend();
  
    void checkpoint( const std::string &label, int version,
                     const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views );
  
    bool restart_available( const std::string &label, int version );
    int latest_version (const std::string &label) const noexcept;
  
    void restart( const std::string &label, int version,
                  const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views );

    void clear_checkpoints();
  
    void register_hashes( const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views,
      const std::vector< Detail::CrefImpl > &crefs );

    void reset();

  private:
    
    std::unordered_map< Detail::MemProtectKey, Detail::MemProtectBlock > m_cref_registry;
    std::unordered_map< Detail::MemProtectKey, Detail::MemProtectBlock > m_view_registry;
    
    MPI_Comm m_mpi_comm;
    context_type *m_context;
    
    mutable int m_latest_version;
  };
  
  class VeloCFileBackend
  {
  public:
  
    VeloCFileBackend( MPIContext< VeloCFileBackend > &ctx, MPI_Comm mpi_comm, const std::string &veloc_config);
    ~VeloCFileBackend();
  
    void checkpoint( const std::string &label, int version,
                     const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views );
  
    bool restart_available( const std::string &label, int version );
    int latest_version (const std::string &label) const noexcept;
  
    void restart( const std::string &label, int version,
                  const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > &views );
  
    void register_hashes( const std::vector< std::unique_ptr< Kokkos::ViewHolderBase > > & ) {} // Do nothing

  private:
  
    MPI_Comm m_mpi_comm;
    MPIContext< VeloCFileBackend > *m_context;
  };
}

#endif  // INC_RESILIENCE_VELOC_VELOCBACKEND_HPP
