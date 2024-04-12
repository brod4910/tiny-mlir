#ifndef TINY_DIALECT
#define TINY_DIALECT

include "mlir/IR/DialectBase.td"

def Tiny_Dialect : Dialect {
  // The name of the dialect.
  let name = "tiny";

  // Short summary of the dialect.
  let summary = "An MLIR for tiny ml-ops.";

  // The description of the dialect.
  let description = [{
        An MLIR modeled after the tinygrad repo. The layout of operations/specs/functionality 
        will be similar to tinygrad or model it closely. 
        Taken from the repo:

        Low-Level Ops (LLOps):
            Buffer                                                       # class of memory on this device
            unary_op  (NOOP, EXP2, LOG2, CAST, SIN, SQRT)                # A -> A
            reduce_op (SUM, MAX)                                         # A -> B (smaller size, B has 1 in shape)
            binary_op (ADD, SUB, MUL, DIV, CMPEQ, MAX)                   # A + A -> A (all the same size)
            load_op   (EMPTY, CONST, FROM, CONTIGUOUS, CUSTOM)           # -> A   (initialize data on device)
            ternary_op (WHERE)                                           # A, A, A -> A

        Mid-Level Ops (MLOps):
            Relu, Log, Exp, Sin                            # unary ops
            Sum, Max                                       # reduce ops (with axis argument)
            Maximum, Add, Sub, Mul, Pow, Div, Equal        # binary ops (no broadcasting, use expand)
            Expand, Reshape, Permute, Pad, Shrink, Flip    # movement ops
            Where                                          # ternary ops

        High-Level Ops (HLOps):
            These are the syntax sugar. They are built on top of the mlops and support most of the things that you could expect from a tensor library.
    }];

    let cppNamespace = "::mlir::tiny";

    let hasCanonicalizer = 1;
}
#endif // TINY_DIALECT