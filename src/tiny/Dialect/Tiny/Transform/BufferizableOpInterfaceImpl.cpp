#include "tiny/Dialect/Tiny/Transform/BufferizableOpInterfaceImpl.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace mlir::tiny {
namespace {
/*
ElementwiseMappable ops don't require bufferization interface implementation
since they get converted through the -convert-elementwise-to-linalg.

Elementwise Mappable Ops:
    * Exp2Op
    * Log2Op
    * SinOp
    * SqrtOp
    * NegOp
    * RecipOp


*/

} // namespace
} // namespace mlir::tiny

void tiny::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registery) {}