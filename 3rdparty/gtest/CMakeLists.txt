set(target "gtest")

file(GLOB hdrs "*.h")
file(GLOB srcs "*.cc")

add_library(${target} STATIC ${srcs})

if((CXX_GCC OR CXX_CLANG) AND (${CMAKE_SYSTEM_NAME} MATCHES "Linux"))
    set(Pthread "-pthread")
endif()
target_link_libraries(${target} ${Pthread})