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
#include <tecocc.h>

#include <cstdlib>

#include "../../runtime/sdaa/sdaa_common.h"
#include "../../runtime/sdaa/sdaa_module.h"
#include "../build_common.h"
#include "../source/codegen_sdaa.h"

namespace tvm {
namespace codegen {

#define TECOCC_CALL(x)                                                                        \
  {                                                                                          \
    tecoccResult result = x;                                                                  \
    if (result != TECOCC_SUCCESS) {                                                           \
      LOG(FATAL) << "TecoccError: " #x " failed with error: " << tecoccGetErrorString(result); \
    }                                                                                        \
  }

// sdaa header files 
// It depends on whether more data types are needed, where is refered in codegen_sdaa.h
std::string FindSDAAIncludePath() {
#if defined(_WIN32)
  const std::string delimiter = "\\";
#else
  const std::string delimiter = "/";
#endif
  std::string sdaa_include_path;
  const char* sdaa_path_env = std::getenv("SDAA_PATH");
  if (sdaa_include_path != nullptr) {
    sdaa_include_path += sdaa_path_env;
    sdaa_include_path += delimiter + "include";
    return sdaa_include_path;
  }

#if defined(__linux__)
  struct stat st;
  sdaa_include_path = "/usr/local/cuda/include";
  if (stat(cuda_include_path.c_str(), &st) == 0) {
    return cuda_include_path;
  }

  if (stat("/usr/include/cuda.h", &st) == 0) {
    return "/usr/include";
  }
#endif
  LOG(FATAL) << "Cannot find cuda include path."
             << "CUDA_PATH is not set or CUDA is not installed in the default installation path."
             << "In other than linux, it is necessary to set CUDA_PATH.";
  return cuda_include_path;
}

std::string TECOCCCompile(const std::string& code, bool include_path = false) {
  std::vector<std::string> compile_params;
  std::vector<const char*> param_cstrings{};
  nvrtcProgram prog;

  // Don't need.
  // std::string cc = "30";
  // int major, minor;
  // cudaError_t e1 = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  // cudaError_t e2 = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);

  // if (e1 == cudaSuccess && e2 == cudaSuccess) {
  //   cc = std::to_string(major) + std::to_string(minor);
  // } else {
  //   LOG(WARNING) << "cannot detect compute capability from your device, "
  //                << "fall back to compute_30.";
  // }

  // compile_params.push_back("-arch=compute_" + cc);

  // if (include_path) {
  //   std::string include_option = "--include-path=" + FindCUDAIncludePath();

  //   compile_params.push_back(include_option);
  // }


// zly: Refer to SDAA C programming guide 4.4.2
  for (const auto& string : compile_params) {
    param_cstrings.push_back(string.c_str());
  }

  NVRTC_CALL(nvrtcCreateProgram(&prog, code.c_str(), nullptr, 0, nullptr, nullptr));
  nvrtcResult compile_res = nvrtcCompileProgram(prog, param_cstrings.size(), param_cstrings.data());

  size_t log_size;
  NVRTC_CALL(nvrtcGetProgramLogSize(prog, &log_size));
  std::string log;
  log.resize(log_size);
  NVRTC_CALL(nvrtcGetProgramLog(prog, &log[0]));
  ICHECK_EQ(compile_res, NVRTC_SUCCESS) << log;
  size_t ptx_size;
  NVRTC_CALL(nvrtcGetPTXSize(prog, &ptx_size));

  std::string ptx;
  ptx.resize(ptx_size);
  NVRTC_CALL(nvrtcGetPTX(prog, &ptx[0]));
  NVRTC_CALL(nvrtcDestroyProgram(&prog));

  return ptx;
}

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
  std::string fmt = "ptx";
  std::string ptx;
  
  //zly: similar to "with Target:" in Python
  const auto* f_enter = Registry::Get("target.TargetEnterScope");
  (*f_enter)(target);
  if (const auto* f = Registry::Get("tvm_callback_cuda_compile")) {
    ptx = (*f)(code).operator std::string();
    // Dirty matching to check PTX vs cubin.
    // TODO(tqchen) more reliable checks
    if (ptx[0] != '/') fmt = "cubin";
  } else {
    ptx = NVRTCCompile(code, cg.need_include_path());
  }
  const auto* f_exit = Registry::Get("target.TargetExitScope");
  (*f_exit)(target);

  return SDAAModuleCreate(ptx, fmt, ExtractFuncInfo(mod), code);
}

TVM_REGISTER_GLOBAL("target.build.sdaa").set_body_typed(BuildSDAA);
}  // namespace codegen
}  // namespace tvm