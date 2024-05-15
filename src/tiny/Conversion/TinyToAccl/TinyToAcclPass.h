#pragma once

#include "mlir/Pass/Pass.h"

#include <memory>
#include <string>

namespace mlir {
class ModuleOp;

namespace tiny {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTinyToAccl();

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTinyToAccl(const std::string target, int numWarps = 4,
                        int threadsPerWarp = 32);

} // namespace tiny
} // namespace mlir