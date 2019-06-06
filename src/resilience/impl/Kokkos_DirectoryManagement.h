#ifndef __KOKKOS_DIRECTORY_MANAGER__
#define __KOKKOS_DIRECTORY_MANAGER__

#include <errno.h>
#include <string>
#include <sys/stat.h>
#include <sstream>
#if !defined( _WIN32 )
  #include <time.h>
#endif

namespace Kokkos {
namespace Experimental {


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
            if ( nErr != EINPROGRESS )
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
               if ( nErr != EINPROGRESS )
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
            if ( nErr != EINPROGRESS )
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

} // Kokkos

} // EXPERIMENTAL
#endif
