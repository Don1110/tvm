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
 *  Build sdaa modules from source.
 *  requires sdaa to be available.
 *
 * \file build_sdaa.cc
 */
#if defined(__linux__)
#include <sys/stat.h>
#endif
#include <sdaa_runtime.h>

#include <cstdlib>

#include "../../runtime/sdaa/sdaa_common.h"
#include "../../runtime/sdaa/sdaa_module.h"
#include "../build_common.h"
#include "../source/codegen_sdaa.h"

namespace tvm {
namespace codegen {


// sdaa header files 
// It depends on whether more data types are needed, where is refered in codegen_sdaa.h
// std::string FindSDAAIncludePath() {
// #if defined(_WIN32)
//   const std::string delimiter = "\\";
// #else
//   const std::string delimiter = "/";
// #endif
//   std::string sdaa_include_path;
//   const char* sdaa_path_env = std::getenv("SDAA_PATH");
//   if (sdaa_include_path != nullptr) {
//     sdaa_include_path += sdaa_path_env;
//     sdaa_include_path += delimiter + "include";
//     return sdaa_include_path;
//   }

// #if defined(__linux__)
//   struct stat st;
//   sdaa_include_path = "/usr/local/cuda/include";
//   if (stat(cuda_include_path.c_str(), &st) == 0) {
//     return cuda_include_path;
//   }

//   if (stat("/usr/include/cuda.h", &st) == 0) {
//     return "/usr/include";
//   }
// #endif
//   LOG(FATAL) << "Cannot find cuda include path."
//              << "CUDA_PATH is not set or CUDA is not installed in the default installation path."
//              << "In other than linux, it is necessary to set CUDA_PATH.";
//   return cuda_include_path;
// }

runtime::Module BuildSDAA(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenSDAA cg;
  cg.Init(output_ssa);

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodeGenSDAA: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenSDAA: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    cg.AddFunction(f);
  }

  std::string code = cg.Finish();

  //zly: if save the cuda code locally or not 
  if (const auto* f = Registry::Get("tvm_callback_cuda_postproc")) {
    code = (*f)(code).operator std::string();
  }
  std::string fmt = "fatbin";
  std::string fatbin;
  
  //zly: similar to "with Target:" in Python
  const auto* f_enter = Registry::Get("target.TargetEnterScope");
  (*f_enter)(target);

  const auto* f = Registry::Get("tvm_callback_sdaa_compile")
  fatbin = (*f)(code).operator std::string();
  
  const auto* f_exit = Registry::Get("target.TargetExitScope");
  (*f_exit)(target);

  return SDAAModuleCreate(fatbin, fmt, ExtractFuncInfo(mod), code);
}

TVM_REGISTER_GLOBAL("target.build.sdaa").set_body_typed(BuildSDAA);
}  // namespace codegen
}  // namespace tvm