#ifndef INC_RESILIENCE_VELOC_VELOCBACKEND_HPP
#define INC_RESILIENCE_VELOC_VELOCBACKEND_HPP

#include <string>
#include <vector>
#include <memory>
#include <Kokkos_Core.hpp>
#include <View/Hooks/Kokkos_ViewHooks.hpp>
#include <unordered_map>
#include <unordered_set>
#include <mpi.h>
#include "../Cref.hpp"

namespace KokkosResilience
{
  class ContextBase;

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

      int id;
      std::vector< unsigned char > buff;
      void *ptr = nullptr;
      std::size_t size = 0;
      std::size_t element_size = 0;
      bool protect = false;
      bool registered = false;
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

    VeloCMemoryBackend( ContextBase &ctx, MPI_Comm mpi_comm );
    ~VeloCMemoryBackend();

    VeloCMemoryBackend( const VeloCMemoryBackend & ) = delete;
    VeloCMemoryBackend( VeloCMemoryBackend && ) noexcept = default;

    VeloCMemoryBackend &operator=( const VeloCMemoryBackend & ) = delete;
    VeloCMemoryBackend &operator=( VeloCMemoryBackend && ) = default;
  
    void checkpoint( const std::string &label, int version,
                     const std::vector< Kokkos::Experimental::ViewHolder > &views );
  
    bool restart_available( const std::string &label, int version );
    int latest_version (const std::string &label) const noexcept;
  
    void restart( const std::string &label, int version,
                  const std::vector< Kokkos::Experimental::ViewHolder > &views );

    void clear_checkpoints();
  
    void register_hashes( const std::vector< Kokkos::Experimental::ViewHolder > &views,
      const std::vector< Detail::CrefImpl > &crefs );

    void reset();
    void register_alias( const std::string &original, const std::string &alias );

  private:

    std::string get_canonical_label( const std::string &_label ) const noexcept;
    
    std::unordered_map< std::string, Detail::MemProtectBlock > m_registry;
    
    MPI_Comm m_mpi_comm;
    ContextBase *m_context;
    
    mutable std::unordered_map< std::string, int > m_latest_version;
    std::unordered_map< std::string, std::string > m_alias_map;
    int m_last_id;
  };

  class VeloCRegisterOnlyBackend : public VeloCMemoryBackend
  {
   public:

    using VeloCMemoryBackend::VeloCMemoryBackend;
    ~VeloCRegisterOnlyBackend() = default;

    VeloCRegisterOnlyBackend( const VeloCRegisterOnlyBackend & ) = delete;
    VeloCRegisterOnlyBackend( VeloCRegisterOnlyBackend && ) noexcept = default;

    VeloCRegisterOnlyBackend &operator=( const VeloCRegisterOnlyBackend & ) = delete;
    VeloCRegisterOnlyBackend &operator=( VeloCRegisterOnlyBackend && ) = default;

    void checkpoint( const std::string &label, int version,
                     const std::vector< Kokkos::Experimental::ViewHolder > &views );

    void restart( const std::string &label, int version,
                  const std::vector< Kokkos::Experimental::ViewHolder > &views );
  };
  
  class VeloCFileBackend
  {
  public:
  
    VeloCFileBackend( MPIContext< VeloCFileBackend > &ctx, MPI_Comm mpi_comm, const std::string &veloc_config);
    ~VeloCFileBackend();
  
    void checkpoint( const std::string &label, int version,
                     const std::vector< Kokkos::Experimental::ViewHolder > &views );

    bool restart_available( const std::string &label, int version );
    int latest_version (const std::string &label) const noexcept;
  
    void restart( const std::string &label, int version,
                  const std::vector< Kokkos::Experimental::ViewHolder > &views );
  
    void register_hashes( const std::vector< Kokkos::Experimental::ViewHolder > & ) {} // Do nothing

  private:
  
    MPI_Comm m_mpi_comm;
    MPIContext< VeloCFileBackend > *m_context;
  };
}

#endif  // INC_RESILIENCE_VELOC_VELOCBACKEND_HPP
