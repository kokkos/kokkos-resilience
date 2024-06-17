/*
 *
 *                        Kokkos v. 3.0
 *       Copyright (2020) National Technology & Engineering
 *               Solutions of Sandia, LLC (NTESS).
 *
 * Under the terms of Contract DE-NA0003525 with NTESS,
 * the U.S. Government retains certain rights in this software.
 *
 * Kokkos is licensed under 3-clause BSD terms of use:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the Corporation nor the names of the
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Questions? Contact Christian R. Trott (crtrott@sandia.gov)
 */

#include "Kokkos_Core.hpp"
#include "ExternalIOInterface.hpp"

namespace KokkosResilience {

   std::string KokkosIOAccessor::resolve_path( std::string path, std::string default_ ) {

      std::string sFullPath = default_;
      size_t pos = path.find("/");
      if ( pos >= 0 && pos < path.length() ) {    // only use the default if there is no path info in the path...
         sFullPath = path;
      } else {
         size_t posII = sFullPath.rfind("/");
         if (posII != (sFullPath.length()-1)) {
            sFullPath += (std::string)"/";
         }
         sFullPath += path;
      }

      return sFullPath;

   }

   // Create empty file...to prevent file access issues when writing / reading in parallel
   void KokkosIOAccessor::create_empty_file ( void * dst )  {

      Kokkos::Impl::SharedAllocationHeader * pData = reinterpret_cast<Kokkos::Impl::SharedAllocationHeader*>(dst);
      KokkosIOInterface * pDataII = reinterpret_cast<KokkosIOInterface*>(pData-1);
      KokkosResilience::KokkosIOAccessor * pAcc = pDataII->pAcc;

      if (pAcc) {
         //printf("calling openfile ...\n");
         pAcc->OpenFile();   // virtual method implemented by specific IO interface
      }
   }

   // Copy from host memory space to designated IO buffer (dst is an instance of KokkosIOAccessor offset by SharedAllocationHeader)
   //                                                      src is the data() pointer from the souce view.
   void KokkosIOAccessor::transfer_from_host ( void * dst, const void * src, size_t size_ )  {

      Kokkos::Impl::SharedAllocationHeader * pData = reinterpret_cast<Kokkos::Impl::SharedAllocationHeader*>(dst);
      KokkosIOInterface * pDataII = reinterpret_cast<KokkosIOInterface*>(pData-1);
      KokkosResilience::KokkosIOAccessor * pAcc = pDataII->pAcc;

      if (pAcc) {
         pAcc->WriteFile( src, size_ );   // virtual method implemented by specific IO interface
      }
   }
   

   // Copy from IO buffer to host memory space  (dst is the data() pointer from the target view
   //                                            src is an instance of KokkosIOAccessor offset by SharedAllocationHeader)
   void KokkosIOAccessor::transfer_to_host ( void * dst, const void * src, size_t size_ ) {

      const Kokkos::Impl::SharedAllocationHeader * pData = reinterpret_cast<const Kokkos::Impl::SharedAllocationHeader*>(src);
      const KokkosIOInterface * pDataII = reinterpret_cast<const KokkosIOInterface*>(pData-1);
      KokkosResilience::KokkosIOAccessor * pAcc = pDataII->pAcc;
      if (pAcc) {
         pAcc->ReadFile( dst, size_ );
      }
   }

   void KokkosIOConfigurationManager::load_configuration ( std::string path ) {

      if (path.length() == 0 ) {
         printf("WARNING:KR_IO_CONFIG not set. loading default setting for HDF5 files access. \n");
         return;
      }

      boost::property_tree::ptree pt;
      boost::property_tree::json_parser::read_json( path, pt );

      for (auto & ar: pt) {
         boost::property_tree::ptree ptII = ar.second;
         std::string name = ptII.get<std::string>("name");
         m_config_list[name] = ptII;
      }

   }

   KokkosIOConfigurationManager * KokkosIOConfigurationManager::get_instance() {
      if (KokkosIOConfigurationManager::m_Inst == nullptr) {
         KokkosIOConfigurationManager::m_Inst = new KokkosIOConfigurationManager;
         std::string path;
         char * config = std::getenv( "KR_IO_CONFIG" );
         if (config != nullptr)
            path = config;
         // printf("loading IOConfigurationManager: %s\n", path.c_str());
         KokkosIOConfigurationManager::m_Inst->load_configuration(path);
      }
      return KokkosIOConfigurationManager::m_Inst;
   }

   KokkosIOConfigurationManager * KokkosIOConfigurationManager::m_Inst = nullptr;

} // namespace KokkosResilience

