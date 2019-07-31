/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/
#include "Kokkos_Core.hpp"
#include "StdFileSpace.hpp"
#include "sys/stat.h"

namespace KokkosResilience {

   int KokkosStdFileAccessor::initialize( const std::string & filepath ) {

       file_path = filepath;
       return 0;

   }

   size_t KokkosStdFileAccessor::OpenFile_impl() {

      open_file(KokkosIOAccessor::WRITE_FILE);
      close_file();
      return 0;
   }


   bool KokkosStdFileAccessor::open_file( int read_write ) {
      
      // printf("open_file: %s, %d\n", file_path.c_str(), read_write );
      if (file_strm.is_open()) {
         printf("file was left open...closing\n");
         close_file();
      }
      std::string sFullPath = KokkosResilience::StdFileSpace::s_default_path;
      size_t pos = file_path.find("/");
      size_t pos2 = file_path.find("./");
      if ( ((int)pos == 0) || ((int)pos2 == 0) ) {    // only use the default if there is no resolveable path...
         sFullPath = file_path;
      } else {
         // printf("building file path %s, %s \n", sFullPath.c_str(), file_path.c_str() );
         sFullPath += (std::string)"/";
         sFullPath += file_path;
      }
      // printf("opening file: %s, %d\n", sFullPath.c_str(), read_write );

       if ( read_write == KokkosIOAccessor::WRITE_FILE ) {
            file_strm.open( sFullPath.c_str(), std::ios::out | std::ios::trunc | std::ios::binary );
       } else if (read_write == KokkosIOAccessor::READ_FILE ) {
            file_strm.open( sFullPath.c_str(), std::ios::in | std::ios::binary );
       } else {
            printf("open_file: incorrect read write parameter specified .\n");
            return -1;
       }

      return file_strm.is_open();

   }

   size_t KokkosStdFileAccessor::ReadFile_impl(void * dest, const size_t dest_size) {
      size_t dataRead = 0;
      char* ptr = (char*)dest;
      if (open_file(KokkosIOAccessor::READ_FILE)) {
         // printf("reading file: %08x, %ld \n", (unsigned long)dest, dest_size);
         while ( !file_strm.eof() && dataRead < dest_size ) {
            file_strm.read( &ptr[dataRead], dest_size );
            dataRead += file_strm.gcount();
         }
      } else {
         printf("WARNING: cannot open file for reading: %s\n", file_path.c_str());
      }
      close_file();
      if (dataRead < dest_size) {
         printf("StdFile: less data available than requested \n");
      }
      return dataRead;

   }
   
   size_t KokkosStdFileAccessor::WriteFile_impl(const void * src, const size_t src_size) {
      size_t m_written = 0;
      char* ptr = (char*)src;
      if (open_file(KokkosIOAccessor::WRITE_FILE) ) {
          file_strm.write(&ptr[0], src_size);
          if (!file_strm.fail())
             m_written = src_size;
      }
      close_file();
      if (m_written != src_size) {
         printf("StdFile: write failed \n");
      }
      return m_written;
   }
   void KokkosStdFileAccessor::close_file() {
      if (file_strm.is_open()) {
         file_strm.close();
      }
   }

   void KokkosStdFileAccessor::finalize() {
      close_file();
   }

   std::string StdFileSpace::s_default_path = "./";

   StdFileSpace::StdFileSpace() {

   }

   /**\brief  Allocate untracked memory in the space */
   void * StdFileSpace::allocate( const size_t arg_alloc_size, const std::string & path ) const {
      KokkosStdFileAccessor * pAcc = new KokkosStdFileAccessor( arg_alloc_size, path );
      pAcc->initialize( path );
      KokkosIOInterface * pInt = new KokkosIOInterface;
      pInt->pAcc = pAcc;
      // printf("allocate std file: %s, %08x \n", path.c_str(), (unsigned long)pAcc);
      return (void*)pInt;

   }

   /**\brief  Deallocate untracked memory in the space */
   void StdFileSpace::deallocate( void * const arg_alloc_ptr
                             , const size_t arg_alloc_size ) const {
       const KokkosIOInterface * pInt = reinterpret_cast<KokkosIOInterface *>(arg_alloc_ptr);
       if (pInt) {
          KokkosStdFileAccessor * pAcc = static_cast<KokkosStdFileAccessor*>(pInt->pAcc);

          if (pAcc) {
             pAcc->finalize();
             delete pAcc;
          }

          delete pInt;

       }

   }
  
   void StdFileSpace::restore_all_views() {
      typedef Kokkos::Impl::SharedAllocationRecord<void,void> base_record;
      Kokkos::Impl::MirrorTracker * pList = base_record::get_filtered_mirror_list( (std::string)name() );
      if (pList == nullptr)  printf("%s::restore views mirror list returned empty list \n", name());
      while (pList != nullptr) {
         Kokkos::Impl::DeepCopy< Kokkos::HostSpace, KokkosResilience::StdFileSpace, Kokkos::DefaultHostExecutionSpace >
                        (((base_record*)pList->src)->data(), ((base_record*)pList->dst)->data(), ((base_record*)pList->src)->size());
         // delete the records along the way...
         if (pList->pNext == nullptr) {
            delete pList;
            pList = nullptr;
         } else {
            // printf("restore next record: %08x \n", (unsigned long)pList->pNext);
            pList = pList->pNext;
            // printf("record: %08x, %08x \n", (unsigned long)pList->src, (unsigned long)pList->dst);
            if (pList->pPrev != nullptr) delete pList->pPrev;
         }
      }
   }
   
   void StdFileSpace::restore_view(const std::string lbl) {
      typedef Kokkos::Impl::SharedAllocationRecord<void,void> base_record;
      Kokkos::Impl::MirrorTracker * pRes = base_record::get_filtered_mirror_entry( (std::string)name(), lbl );
      if (pRes != nullptr) {
         Kokkos::Impl::DeepCopy< Kokkos::HostSpace, KokkosResilience::StdFileSpace, Kokkos::DefaultHostExecutionSpace >
                        (((base_record*)pRes->src)->data(), ((base_record*)pRes->dst)->data(), ((base_record*)pRes->src)->size());
         delete pRes;
      }
   }
  
   void StdFileSpace::checkpoint_create_view_targets() {
      typedef Kokkos::Impl::SharedAllocationRecord<void,void> base_record;
      Kokkos::Impl::MirrorTracker * pList = base_record::get_filtered_mirror_list( (std::string)name() );
      if (pList == nullptr) {
         printf("memspace %s returned empty list of checkpoint views \n", name());
      }
      while (pList != nullptr) {
         KokkosIOAccessor::create_empty_file(((base_record*)pList->dst)->data());
         // delete the records along the way...
         if (pList->pNext == nullptr) {
            delete pList;
            pList = nullptr;
         } else {
            pList = pList->pNext;
            delete pList->pPrev;
         }
      }
      
   }
   void StdFileSpace::checkpoint_views() {
      typedef Kokkos::Impl::SharedAllocationRecord<void,void> base_record;
      Kokkos::Impl::MirrorTracker * pList = base_record::get_filtered_mirror_list( (std::string)name() );
      if (pList == nullptr) {
         printf("memspace %s returned empty list of checkpoint views \n", name());
      }
      while (pList != nullptr) {
 //     typedef Kokkos::Impl::SharedAllocationRecord<void,void> base_record;
         Kokkos::Impl::DeepCopy< KokkosResilience::StdFileSpace, Kokkos::HostSpace, Kokkos::DefaultHostExecutionSpace >
                        (((base_record*)pList->dst)->data(), ((base_record*)pList->src)->data(), ((base_record*)pList->src)->size());
         // delete the records along the way...
         if (pList->pNext == nullptr) {
            delete pList;
            pList = nullptr;
         } else {
            pList = pList->pNext;
            delete pList->pPrev;
         }
      }
      
   }
   void StdFileSpace::set_default_path( const std::string path ) {

      StdFileSpace::s_default_path = path;

   }
  
} // namespace KokkosResilience



namespace Kokkos {
namespace Impl {

#ifdef KOKKOS_DEBUG
SharedAllocationRecord< void , void >
SharedAllocationRecord< KokkosResilience::StdFileSpace , void >::s_root_record ;
#endif

void
SharedAllocationRecord< KokkosResilience::StdFileSpace , void >::
deallocate( SharedAllocationRecord< void , void > * arg_rec )
{
  delete static_cast<SharedAllocationRecord*>(arg_rec);
}

SharedAllocationRecord< KokkosResilience::StdFileSpace , void >::
~SharedAllocationRecord()
{
  #if defined(KOKKOS_ENABLE_PROFILING)
  if(Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::deallocateData(
      Kokkos::Profiling::SpaceHandle(KokkosResilience::StdFileSpace::name()),RecordBase::m_alloc_ptr->m_label,
      data(),size());
  }
  #endif

  m_space.deallocate( SharedAllocationRecord< void , void >::m_alloc_ptr
                    , SharedAllocationRecord< void , void >::m_alloc_size
                    );
}

SharedAllocationRecord< KokkosResilience::StdFileSpace , void >::
SharedAllocationRecord( const KokkosResilience::StdFileSpace & arg_space
                      , const std::string       & arg_label
                      , const size_t              arg_alloc_size
                      , const SharedAllocationRecord< void , void >::function_type arg_dealloc
                      )
  // Pass through allocated [ SharedAllocationHeader , user_memory ]
  // Pass through deallocation function
  : SharedAllocationRecord< void , void >
      (
#ifdef KOKKOS_DEBUG
      & SharedAllocationRecord< KokkosResilience::StdFileSpace , void >::s_root_record,
#endif
        reinterpret_cast<SharedAllocationHeader*>( arg_space.allocate( arg_alloc_size, arg_label ) )
      , arg_alloc_size
      , arg_dealloc
      )
  , m_space( arg_space )
{
#if defined(KOKKOS_ENABLE_PROFILING)
  if(Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::allocateData(Kokkos::Profiling::SpaceHandle(arg_space.name()),arg_label,data(),arg_alloc_size);
   }
#endif
  // Fill in the Header information
  RecordBase::m_alloc_ptr->m_record = static_cast< SharedAllocationRecord< void , void > * >( this );

  strncpy( RecordBase::m_alloc_ptr->m_label
          , arg_label.c_str()
          , SharedAllocationHeader::maximum_label_length
          );
  // Set last element zero, in case c_str is too long
  RecordBase::m_alloc_ptr->m_label[SharedAllocationHeader::maximum_label_length - 1] = (char) 0;
}

//----------------------------------------------------------------------------

void * SharedAllocationRecord< KokkosResilience::StdFileSpace , void >::
allocate_tracked( const KokkosResilience::StdFileSpace & arg_space
                , const std::string & arg_alloc_label
                , const size_t arg_alloc_size )
{
  if ( ! arg_alloc_size ) return (void *) 0 ;

  SharedAllocationRecord * const r =
    allocate( arg_space , arg_alloc_label , arg_alloc_size );

  RecordBase::increment( r );

  return r->data();
}

void SharedAllocationRecord< KokkosResilience::StdFileSpace , void >::
deallocate_tracked( void * const arg_alloc_ptr )
{
  if ( arg_alloc_ptr != 0 ) {
    SharedAllocationRecord * const r = get_record( arg_alloc_ptr );

    RecordBase::decrement( r );
  }
}

void * SharedAllocationRecord< KokkosResilience::StdFileSpace , void >::
reallocate_tracked( void * const arg_alloc_ptr
                  , const size_t arg_alloc_size )
{
  SharedAllocationRecord * const r_old = get_record( arg_alloc_ptr );
  SharedAllocationRecord * const r_new = allocate( r_old->m_space , r_old->get_label() , arg_alloc_size );

  RecordBase::increment( r_new );
  RecordBase::decrement( r_old );

  return r_new->data();
}

SharedAllocationRecord< KokkosResilience::StdFileSpace , void > *
SharedAllocationRecord< KokkosResilience::StdFileSpace , void >::get_record( void * alloc_ptr )
{
  typedef SharedAllocationHeader  Header ;
  typedef SharedAllocationRecord< KokkosResilience::StdFileSpace , void >  RecordHost ;

  SharedAllocationHeader const * const head   = alloc_ptr ? Header::get_header( alloc_ptr ) : (SharedAllocationHeader *)0 ;
  RecordHost                   * const record = head ? static_cast< RecordHost * >( head->m_record ) : (RecordHost *) 0 ;

  if ( ! alloc_ptr || record->m_alloc_ptr != head ) {
    Kokkos::Impl::throw_runtime_exception( std::string("Kokkos::Impl::SharedAllocationRecord< KokkosResilience::StdFileSpace , void >::get_record ERROR" ) );
  }

  return record ;
}

// Iterate records to print orphaned memory ...
void SharedAllocationRecord< KokkosResilience::StdFileSpace , void >::
print_records( std::ostream & s , const KokkosResilience::StdFileSpace & , bool detail )
{
#ifdef KOKKOS_DEBUG
  SharedAllocationRecord< void , void >::print_host_accessible_records( s , "StdFileSpace" , & s_root_record , detail );
#else
  throw_runtime_exception("SharedAllocationRecord<StdFileSpace>::print_records only works with KOKKOS_DEBUG enabled");
#endif
}

} // namespace Impl
} // namespace Kokkos

