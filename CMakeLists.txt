cmake_minimum_required(VERSION 3.20.0)

project(tiny LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_DEPS ON)

set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE INTERNAL "" FORCE)
set(CMAKE_OSX_DEPLOYMENT_TARGET "14.4" CACHE INTERNAL "" FORCE)

set(LLVM_EXTERNAL_LIT "/Users/brodriguez/miniconda3/envs/llvm/bin/lit")

set(LLVM_BUILD_DIR "" CACHE PATH "Path to LLVM build directory")
message(STATUS "Using LLVM build directory: ${LLVM_BUILD_DIR}")
set(LLVM_DIR "${LLVM_BUILD_DIR}/lib/cmake/llvm" CACHE PATH "Path to LLVM build directory" FORCE)
set(MLIR_DIR "${LLVM_BUILD_DIR}/lib/cmake/mlir" CACHE PATH "Path to MLIR build directory" FORCE)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})


include(TableGen)
include(AddLLVM)


# Utilities
function(add_tiny_object name)
  cmake_parse_arguments(ARG "" "" "DEPENDS;LINK_LIBS" ${ARGN})
  add_library(${name} OBJECT)
  target_sources(${name}
    PRIVATE ${ARG_UNPARSED_ARGUMENTS}
    INTERFACE $<TARGET_OBJECTS:${name}>
  )


  # add_library(${name} OBJECT ${ARG_UNPARSED_ARGUMENTS})
  if(ARG_DEPENDS)
    add_dependencies(${name} ${ARG_DEPENDS})
  endif()
  if(ARG_LINK_LIBS)
    target_link_libraries(${name} PUBLIC ${ARG_LINK_LIBS})
  endif()
endfunction()

set_property(GLOBAL PROPERTY TINY_LIBS "")
function(add_tiny_library name)
  set_property(GLOBAL APPEND PROPERTY TINY_LIBS ${name})
  add_tiny_object(${name} ${ARGN})
  llvm_update_compile_flags(${name})
endfunction()

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR}/src) # Tablegen'd files

add_subdirectory(tests)
add_subdirectory(tools)
add_subdirectory(src/tiny)