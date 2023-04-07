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

# SDAA Module
find_sdaa(${USE_SDAA})

if(SDAA_FOUND)
  # always set the includedir when sdaa is available
  # avoid global retrigger of cmake
  include_directories(SYSTEM ${SDAA_INCLUDE_DIRS})
endif(SDAA_FOUND)

if(USE_SDAA)
  if(NOT SDAA_FOUND)
    message(FATAL_ERROR "Cannot find SDAA, USE_SDAA=" ${USE_SDAA})
  endif()
  message(STATUS "Build with SDAA ${SDAA_VERSION} support")
  tvm_file_glob(GLOB RUNTIME_SDAA_SRCS src/runtime/sdaa/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_SDAA_SRCS})
  list(APPEND COMPILER_SRCS src/target/opt/build_sdaa_on.cc)

  list(APPEND TVM_LINKER_LIBS ${SDAA_NVRTC_LIBRARY})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${SDAA_SDAART_LIBRARY})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${SDAA_SDAA_LIBRARY})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${SDAA_NVRTC_LIBRARY})

else(USE_SDAA)
  list(APPEND COMPILER_SRCS src/target/opt/build_sdaa_off.cc)
endif(USE_SDAA)
