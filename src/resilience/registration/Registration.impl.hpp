#ifndef _INC_RESILIENCE_REGISTRATION_IMPL_HPP
#define _INC_RESILIENCE_REGISTRATION_IMPL_HPP

#include "Registration.hpp"

#include "resilience/context/ContextBase.hpp"

#include "./ViewHolder.hpp"
#include "./Custom.hpp"

#ifdef KR_ENABLE_MAGISTRATE
#include "./Magistrate.hpp"
#else
#include "./Simple.hpp"
#endif

#endif
