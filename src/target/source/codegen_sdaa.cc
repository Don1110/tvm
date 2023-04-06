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
 * \file codegen_sdaa.cc
 */

#include "codegen_sdaa.h"

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/stmt_functor.h>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace codegen {

CodeGenSDAA::CodeGenSDAA() { 
  //restrict_keyword_ = "__restrict__"; 
}

// zly: currently barrier is not supported in SDAA.
void CodeGenSDAA::Init(bool output_ssa) {
  CodeGenC::Init(output_ssa);
  // vid_global_barrier_state_ = name_supply_->FreshName(runtime::symbol::tvm_global_barrier_state);
  // vid_global_barrier_expect_ = name_supply_->FreshName("__barrier_expect");
  // ICHECK_EQ(vid_global_barrier_state_, runtime::symbol::tvm_global_barrier_state);
}

void CodeGenSDAA::PrintFuncPrefix(std::ostream& os) { os << "extern \"C\" __global__ void"; }

void CodeGenSDAA::VisitStmt_(const tir::ForNode* op) {
  ICHECK(is_const_int(op->min, 0));
  if (op->kind == tir::ForKind::kUnrolled) {
    PrintIndent();
    stream << "#pragma unroll\n";
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenSDAA::BindThreadIndex(const IterVar& iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));
  var_idmap_[iv->var.get()] = "_PEN";
  // std::string thread_tag = "_PEN"; //data type: long unsigned int, 32 bits
  // var_idmap_[iv->var.get()] = CastFromTo(thread_tag, DataType::UInt(32), iv->var.dtype());
}

void CodeGenSDAA::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
  //zly: adapt codegen_c_host::PrintType, as codegen_cuda is too complicated and sdaa cannot support.
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK_EQ(lanes, 1) << "does not support vector types";
    os << "void*";
    return;
  }

  if (t.is_void()) {
    os << "void";
    return;
  }
  if (t == DataType::Bool()) {
    os << "bool";
    return;
  }
  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
      case 16:
        os << "_Float16";
        break;
      case 32:
        os << "float";
        break;
      case 64:
        os << "double";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    // zly: sdaa don't support vector types except one case that lanes == 16
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes;
      return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << 'u';
    }
    switch (t.bits()) {
      case 8:
        os << "int8_t";
        break;
      case 16:
        os << "int16_t";
        break;
      case 32:
        os << "int32_t";
        break;
      case 64:
        os << "int64_t";
        break;
      case 1:
        os << "int32_t";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    // zly: sdaa don't support vector types except one case that lanes == 16
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes;
      return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to sdaa type";
}


// zly: not supported currently
// void CodeGenSDAA::PrintStorageSync(const CallNode* op) {
//   const std::string& sync = op->args[0].as<StringImmNode>()->value;
//   if (sync == "warp") {
//     LOG(FATAL) << "warp barrier not supported";
//   } else if (sync == "shared") {
//     LOG(FATAL) << "shared barrier not supported";
//   } else if (sync == "global") {
//     os << "__device__";
//   }
// }


void CodeGenSDAA::PrintStorageScope(const std::string& scope, std::ostream& os) {  // NOLINT(*)
  if (scope == "global") {
    os << "__device__ ";
  } else if (scope == "shared") {
    // do nothing
  } else {
    LOG(FATAL) << "StorageScope " << scope <<" not supported";
  }
}


//zly: need to further review.
void CodeGenSDAA::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == tir::attr::fragment_shape) {
    const VarNode* buffer = op->node.as<VarNode>();
    const StringImmNode* shape_str = op->value.as<StringImmNode>();
    fragment_shapes[buffer] = shape_str->value;
  } else if (op->attr_key == tir::attr::fragment_layout) {
    const VarNode* buffer = op->node.as<VarNode>();
    const StringImmNode* layout_str = op->value.as<StringImmNode>();
    fragment_layouts[buffer] = layout_str->value;
  } else if (op->attr_key == tir::attr::async_commit_queue_scope) {
    const IntImmNode* queue_id = op->value.as<IntImmNode>();
    ICHECK(queue_id && queue_id->value == 0) << "For SDAA, the index of an async queue must be 0.";
    this->VisitStmt(op->body);
    auto commit_group = Call(DataType::Void(), builtin::ptx_commit_group(), {});
    this->VisitExpr(commit_group, this->stream);
    return;
  } else if (op->attr_key == tir::attr::async_wait_queue_scope) {
    auto wait_attrs = GetAsyncWaitAttributes(op);
    auto queue_id = wait_attrs.first.as<IntImmNode>();
    ICHECK(queue_id && queue_id->value == 0) << "For SDAA, the index of an async queue must be 0.";
    auto wait_cnt = wait_attrs.second;
    auto wait_group = Call(DataType::Void(), builtin::ptx_wait_group(), {wait_cnt});
    this->VisitExpr(wait_group, this->stream);
    auto inner = op->body.as<AttrStmtNode>();
    ICHECK(inner);
    this->VisitStmt(inner->body);
    return;
  }
  CodeGenC::VisitStmt_(op);
}

}  // namespace codegen
}  // namespace tvm
