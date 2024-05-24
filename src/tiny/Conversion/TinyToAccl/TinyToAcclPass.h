#pragma once

#include <memory>
#include <string>

namespace mlir {
class ModuleOp;
template <typename T> class OperationPass;

namespace tiny {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTinyToAcclPass();

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTinyToAcclPass(const std::string target, int numWarps = 4,
                            int threadsPerWarp = 32);

} // namespace tiny
} // namespace mlir