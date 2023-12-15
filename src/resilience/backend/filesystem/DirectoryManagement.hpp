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
#ifndef INC_RESILIENCE_FILESSYSTEM_DIRECTORYMANAGEMENT_HPP
#define INC_RESILIENCE_FILESSYSTEM_DIRECTORYMANAGEMENT_HPP

#include <errno.h>
#include <string>
#include <sys/stat.h>
#include <sstream>
#if !defined( _WIN32 )
  #include <time.h>
#endif

namespace KokkosResilience {

template<class MemorySpace>
struct DirectoryManager {

   template<typename D>
   inline static std::string ensure_directory_exists( bool bCreate, const std::string dir, D d ) {
      //printf("last call creating dir: %s \n", dir.c_str());
      int nErr = 0;
      if (bCreate) {
         for ( int nRetry = 0; nRetry < 5; nRetry++ ) {
            mkdir(dir.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            nErr = errno;
            if ( ( nErr != EINPROGRESS ) && ( nErr != EAGAIN ))
               break;
            thread_yield();
         }
      }
      if( bCreate == false || nErr == EEXIST || nErr == 0 || nErr == 3 ) {
         std::string path = dir;
         std::stringstream iter_num;
         iter_num << "/" << d << "/";
         path += iter_num.str();
        // printf("final dir: %s \n", path.c_str());
         int nErr = 0;
         if (bCreate) {
            for ( int nRetry = 0; nRetry < 5; nRetry++ ) {
               mkdir(path.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
               nErr = errno;
               if ( (nErr != EINPROGRESS) && ( nErr != EAGAIN ) )
                  break;
               thread_yield();
            }
         }
         if( bCreate == false || nErr == EEXIST || nErr == 0 || nErr == 3 ) {
            return path;
         } else {
            printf("WARNING: Error creating path: %s, %d \n", path.c_str(), errno);
            return "";
         }
      } else {
         printf("WARNING: Error creating path: %s, %d \n", dir.c_str(), errno);
         return "";
      }
   }

   template<typename D, typename ...Dargs>
   inline static std::string ensure_directory_exists( bool bCreate, const std::string dir, D d, Dargs... dargs) {
    //  printf("recursive dir call: %s \n", dir.c_str());
      int nErr = 0;
      if (bCreate) {
         for ( int nRetry = 0; nRetry < 5; nRetry++ ) {
            mkdir(dir.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            nErr = errno;
            if ( (nErr != EINPROGRESS) && ( nErr != EAGAIN ) )
               break;
            thread_yield();
         }
      }
      if( bCreate == false || nErr == EEXIST || nErr == 0 || nErr == 3) {
         std::string path = dir;
         std::stringstream iter_num;
         iter_num << "/" << d << "/";
         path += iter_num.str();
         return ensure_directory_exists( bCreate, path, dargs... );
      } else {
         printf("WARNING: Error creating path: %s, %d \n", dir.c_str(), errno);
         return "";
      }
   }

   template<class ... Dargs>
   inline static int set_checkpoint_directory(bool bCreate, std::string dir, Dargs ...dargs ) {
      std::string path = ensure_directory_exists( bCreate, dir, dargs... );
      if ( path.length() > 0 ) {
          MemorySpace::set_default_path(path);
          return 0;
      } else {
         return -1;
      }
   }
   inline static void thread_yield() {

      #if !defined( _WIN32 )
      timespec req ;
      req.tv_sec  = 0 ;
      req.tv_nsec = 4096;
      nanosleep( &req, nullptr );
      #endif
   }
};

} // namespace KokkosResilience

#endif  // INC_RESILIENCE_FILESSYSTEM_DIRECTORYMANAGEMENT_HPP
