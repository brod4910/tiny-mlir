#pragma once

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"

namespace mlir {
class DialectRegistry;

namespace tiny {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace tiny
} // namespace mlir