cmake_minimum_required(VERSION 3.4.3)

project(test_ie CXX)

find_package(InferenceEngine REQUIRED)
# find_package(ngraph REQUIRED)

# add_executable(${CMAKE_PROJECT_NAME} main.cpp)
# target_compile_features(${CMAKE_PROJECT_NAME} PRIVATE cxx_range_for)

# target_link_libraries(${CMAKE_PROJECT_NAME}
#   ${InferenceEngine_LIBRARIES}
# #   ${NGRAPH_LIBRARIES}
# )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

macro(add_test)
    add_executable("${ARGV0}" "${ARGV1}")
    target_include_directories("${ARGV0}" PRIVATE "${PROJECT_SOURCE_DIR}/3rdparty/")
    target_link_libraries("${ARGV0}" PRIVATE gtest pthread)
endmacro()

add_subdirectory(3rdparty/gtest)
add_subdirectory(0_git)
