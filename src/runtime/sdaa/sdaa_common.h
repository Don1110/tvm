/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file cuda_common.h
 * \brief Common utilities for CUDA
 */
#ifndef TVM_RUNTIME_CUDA_CUDA_COMMON_H_
#define TVM_RUNTIME_CUDA_CUDA_COMMON_H_

#include <sdaa.h>
#include <tvm/runtime/packed_func.h>

#include <string>

#include "../workspace_pool.h"

namespace tvm {
namespace runtime {

#define SDAA_DRIVER_CALL(x)                                             \
  {                                                                     \
    SDresult result = x;                                                \
    if (result != SDAA_SUCCESS && result != SDAA_ERROR_DEINITIALIZED) { \
      const char* msg;                                                  \
      sdaaGetErrorName(result, &msg);                                     \
      LOG(FATAL) << "SDAAError: " #x " failed with error: " << msg;     \
    }                                                                   \
  }

#define SDAA_CALL(func)                                       \
  {                                                           \
    sdaaError_t e = (func);                                   \
    ICHECK(e == sdaaSuccess || e == sdaaErrorSdaartUnloading) \
        << "SDAA: " << sdaaGetErrorString(e);                 \
  }

/*! \brief Thread local workspace */
class SDAAThreadEntry {
 public:
  /*! \brief The sdaa stream */
  sdaaStream_t stream{0};
  /*! \brief thread local pool*/
  WorkspacePool pool;
  /*! \brief constructor */
  SDAAThreadEntry();
  // get the threadlocal workspace
  static SDAAThreadEntry* ThreadLocal();
};
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_CUDA_CUDA_COMMON_H_
