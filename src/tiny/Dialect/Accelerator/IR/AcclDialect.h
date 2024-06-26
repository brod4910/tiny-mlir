#pragma once
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

#include "tiny/Dialect/Accelerator/IR/AcclDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "tiny/Dialect/Accelerator/IR/AcclAttrs.h.inc"

namespace mlir::tiny::accl {
/*
Gets the default CTA layout for a particular tensor.

A default CTA Layout is one where each thread handles a single item
in a Tensor.
*/
CTALayoutAttr getDefaultCTALayout(MLIRContext *context,
                                  ArrayRef<int64_t> shape);

/*
Gets the default MMA Encoding for a particular MMA op.

A default MMA encoding is one where each thread handles a single MMA op.

single MMA Op: c = a * b + c

*/
Attribute getDefaultMMAEncoding(MLIRContext *context, int numWarps,
                                int threadsPerWarp);

// TODO: Implement logic to check if broadcasting would be inefficient.
//  Create a pattern that checks and rewrites with perhaps an outer-loop (NP)
LogicalResult isInefficientBroadcasting();

} // namespace mlir::tiny::accl