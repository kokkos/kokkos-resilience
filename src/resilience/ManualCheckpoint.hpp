#ifndef INC_RESILIENCE_MANUALCHECKPOINT_HPP
#define INC_RESILIENCE_MANUALCHECKPOINT_HPP

#ifdef KOKKOS_ENABLE_MANUAL_CHECKPOINT
   #include "resilience/stdio/StdFileSpace.hpp"

   #ifdef KR_ENABLE_HDF5
      #include "hdf5/HDF5Space.hpp"
   #endif
#endif

#endif  // INC_RESILIENCE_MANUALCHECKPOINT_HPP
