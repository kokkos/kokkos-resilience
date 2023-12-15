#ifndef _INC_RESILIENCE_REGISTRATION_HEADERS_HPP
#define _INC_RESILIENCE_REGISTRATION_HEADERS_HPP

#include "./Registration.hpp"

#include "./ViewHolder.hpp"
#include "./Custom.hpp"

#ifdef KR_ENABLE_MAGISTRATE
#include "./Magistrate.hpp"
#else
#include "./Simple.hpp"
#endif

#ifdef KR_ENABLE_VT
#include "./VTProxy.hpp"
#endif

#endif
