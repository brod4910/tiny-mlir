cmake_minimum_required(VERSION 3.20.0)

project(tiny LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_DEPS ON)

set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE INTERNAL "" FORCE)
set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE INTERNAL "" FORCE)

set(LLVM_BUILD_DIR "" CACHE PATH "Path to LLVM build directory")
message(STATUS "Using LLVM build directory: ${LLVM_BUILD_DIR}")
set(LLVM_DIR "${LLVM_BUILD_DIR}/lib/cmake/llvm" CACHE PATH "Path to LLVM build directory" FORCE)
set(MLIR_DIR "${LLVM_BUILD_DIR}/lib/cmake/mlir" CACHE PATH "Path to MLIR build directory" FORCE)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

include(AddLLVM)
include(TableGen)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/externals/llvm-project)
include_directories(${PROJECT_BINARY_DIR})

# LDFLAGS
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/usr/local/opt/llvm/lib/c++ -Wl,-rpath,/usr/local/opt/llvm/lib/c++")

add_subdirectory(tests)
add_subdirectory(tools)
add_subdirectory(src)