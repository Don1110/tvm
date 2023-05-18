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

#include <unistd.h>

#include "../../runtime/sdaa/sdaa_common.h"
#include "../../runtime/sdaa/sdaa_module.h"
#include "../build_common.h"
#include "../source/codegen_sdaa.h"



namespace tvm {
namespace codegen {


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
  if (const auto* f = Registry::Get("tvm_callback_sdaa_postproc")) {
    code = (*f)(code).operator std::string();
  }
  std::string fmt = "so";
  std::string compile_result;
  
  //zly: similar to "with Target:" in Python
  const auto* f_enter = Registry::Get("target.TargetEnterScope");
  (*f_enter)(target);

  const auto* f = Registry::Get("tvm_callback_sdaa_compile");
  std::string ops = "--sdaa-device-only";
  compile_result = (*f)(code, fmt, ops).operator std::string();

  std::ofstream bytecode;
  bytecode.open("bytecode.so", std::ios::out | std::ios::trunc);

  bytecode << compile_result << std::endl;

  bytecode.close();

  const auto* f_exit = Registry::Get("target.TargetExitScope");
  (*f_exit)(target);


  char *path_current;
  path_current = get_current_dir_name();
   
  std::string so_path;
  so_path = path_current;
  so_path.append("/bytecode.so");

  return SDAAModuleCreate(so_path, fmt, ExtractFuncInfo(mod), code);
}

TVM_REGISTER_GLOBAL("target.build.sdaa").set_body_typed(BuildSDAA);
}  // namespace codegen
}  // namespace tvm