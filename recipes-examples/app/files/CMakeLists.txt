# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.13)

# Project
project(onnxruntime_samples C CXX)
if (WIN32)
 string(APPEND CMAKE_CXX_FLAGS " /W4")
else()
 string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
 string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
endif()

#onnxruntime providers
option(LIBPNG_ROOTDIR "libpng root dir")
option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")

set(CMAKE_CXX_STANDARD 17)

# if(NOT ONNXRUNTIME_ROOTDIR)
#   if(WIN32)
#     set(ONNXRUNTIME_ROOTDIR "C:/Program Files/onnxruntime")
# else()
#     set(ONNXRUNTIME_ROOTDIR "/usr/local")
#   endif()
# endif()

# The ORT package has a different include directory structure to a local install via cmake.
# We added the path for the pre-built package above. Add the path for a local install to support either usage.
# TODO: If we want to support additional EPs being loadable from a local install we also need to add EP specific
# directories under /include/onnxruntime/core/providers
# include_directories("${ONNXRUNTIME_ROOTDIR}/include"                           # Pre-built package
#                     "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"               # Linux local install to /usr/local
#                     "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session") # Windows local install

# link_directories("${ONNXRUNTIME_ROOTDIR}/lib")


# On Linux the samples use libjpeg and libpng for decoding images.
# On Windows they use Windows Image Component(WIC)
# if(NOT WIN32)
#     find_package(JPEG)
#     if(LIBPNG_ROOTDIR)
#       set(PNG_FOUND true)
#       set(PNG_LIBRARIES png16)
#       set(PNG_INCLUDE_DIRS "${LIBPNG_ROOTDIR}/include")
#       set(PNG_LIBDIR "${LIBPNG_ROOTDIR}/lib")
#     else()
#       find_package(PNG)
#     endif()
# endif()

add_executable(onnx-test "test.cpp")

target_link_libraries(onnx-test PRIVATE onnxruntime)

install(TARGETS onnx-test DESTINATION bin)
