import tvm
from tvm import te
from tvm.ir.module import IRModule


import numpy as np

n = 1024

#target = "sdaa"

target = tvm.target.Target(target="sdaa", host="llvm")

dev = tvm.device(target.kind.name, 0)

# declare the computation using the expression API
A = te.placeholder((n, ), name="A")
B = te.placeholder((n, ), name="B")
C = te.compute((n,), lambda i: A[i] + B[i], name="C")

# Default schedule
func = te.create_prim_func([A, B, C])
func = func.with_attr("global_symbol", "main")
ir_module = IRModule({"main": func})
print(ir_module.script())

# Construct a TensorIR schedule class from an IRModule
sch = tvm.tir.Schedule(ir_module) 
# Get block by its name
block_c = sch.get_block("C")
# Get loops surronding the block
(i,) = sch.get_loops(block_c)

sch.bind(i, "threadIdx.x")
print(sch.mod.script())

fadd = tvm.build(sch.mod, target=target)
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev) #在目标设备分配相应空间，并将数据填入。
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
fadd(a, b, c)
output = c.numpy()

evaluator = fadd.time_evaluator(fadd.entry_name, dev, number=1)
print("Baseline: %f" % evaluator(a, b, c).mean)
