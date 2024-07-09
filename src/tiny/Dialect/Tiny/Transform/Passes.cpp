#include "tiny/Dialect/Tiny/Transform/Passes.h"
#include "tiny/Dialect/Tiny/Transform/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tiny {
#define GEN_PASS_DEF_TINYBUFFERIZEPASS
#include "tiny/Dialect/Tiny/Transform/Passes.h.inc"
} // namespace tiny
} // namespace mlir

using namespace mlir;
using namespace bufferization;

namespace {
struct TinyBufferizePass
    : public tiny::impl::TinyBufferizePassBase<TinyBufferizePass> {

  void runOnOperation() override {}
};
} // namespace
