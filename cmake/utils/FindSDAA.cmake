# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

#######################################################
# Enhanced version of find SDAA.
#
# Usage:
#   find_sdaa(${USE_SDAA} ${USE_CUDNN})
#
# - When USE_SDAA=ON, use auto search
# - When USE_SDAA=/path/to/sdaa-path, use the sdaa path

#
# Provide variables:
#
# - SDAA_FOUND
# - SDAA_INCLUDE_DIRS

macro(find_sdaa use_sdaa)
  set(__use_sdaa ${use_sdaa})
  if(${__use_sdaa} MATCHES ${IS_TRUE_PATTERN})
    find_package(SDAA QUIET)
  elseif(IS_DIRECTORY ${__use_sdaa})
    set(SDAA_TOOLKIT_ROOT_DIR ${__use_sdaa})
    message(STATUS "Custom SDAA_PATH=" ${SDAA_TOOLKIT_ROOT_DIR})
    set(SDAA_INCLUDE_DIRS ${SDAA_TOOLKIT_ROOT_DIR}/include)
    set(SDAA_FOUND TRUE)
    if(MSVC)
      find_library(SDAA_SDAART_LIBRARY sdaart
        ${SDAA_TOOLKIT_ROOT_DIR}/lib/x64
        ${SDAA_TOOLKIT_ROOT_DIR}/lib/Win32)
    else(MSVC)
      find_library(SDAA_SDAART_LIBRARY sdaart
        ${SDAA_TOOLKIT_ROOT_DIR}/lib64
        ${SDAA_TOOLKIT_ROOT_DIR}/lib)
    endif(MSVC)
  endif()

  # additional libraries
  if(SDAA_FOUND)
    if(MSVC)
      find_library(SDAA_SDAA_LIBRARY sdaa
        ${SDAA_TOOLKIT_ROOT_DIR}/lib/x64
        ${SDAA_TOOLKIT_ROOT_DIR}/lib/Win32)
    else(MSVC)
      find_library(_SDAA_SDAA_LIBRARY sdaa
        PATHS ${SDAA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib targets/x86_64-linux/lib/stubs lib64/stubs
        NO_DEFAULT_PATH)
      if(_SDAA_SDAA_LIBRARY)
        set(SDAA_SDAA_LIBRARY ${_SDAA_SDAA_LIBRARY})
      endif()
    endif(MSVC)

    message(STATUS "Found SDAA_TOOLKIT_ROOT_DIR=" ${SDAA_TOOLKIT_ROOT_DIR})
    message(STATUS "Found SDAA_SDAA_LIBRARY=" ${SDAA_SDAA_LIBRARY})
    message(STATUS "Found SDAA_SDAART_LIBRARY=" ${SDAA_SDAART_LIBRARY})
    # message(STATUS "Found SDAA_NVRTC_LIBRARY=" ${SDAA_NVRTC_LIBRARY})
    # message(STATUS "Found SDAA_CUDNN_INCLUDE_DIRS=" ${SDAA_CUDNN_INCLUDE_DIRS})
    # message(STATUS "Found SDAA_CUDNN_LIBRARY=" ${SDAA_CUDNN_LIBRARY})
    # message(STATUS "Found SDAA_CUBLAS_LIBRARY=" ${SDAA_CUBLAS_LIBRARY})
    # message(STATUS "Found SDAA_CURAND_LIBRARY=" ${SDAA_CURAND_LIBRARY})
    # message(STATUS "Found SDAA_CUBLASLT_LIBRARY=" ${SDAA_CUBLASLT_LIBRARY})
  endif(SDAA_FOUND)
endmacro(find_sdaa)
