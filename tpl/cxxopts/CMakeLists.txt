include(FetchContent)

FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG v2.2.1
)

FetchContent_MakeAvailable(cxxopts)
# Older versions of cxxopts don't have this
add_library(cxxopts::cxxopts ALIAS cxxopts)