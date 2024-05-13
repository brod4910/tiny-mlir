#pragma once

#include <memory>
namespace mlir {
namespace tiny {
class ModuleOp;
template <typename T> class OperationPass;

std::unique_ptr<OperationPass<ModuleOp>> createConvertTinyToAccl();

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTinyToAccl(const std::string target, int numWarps = 4,
                        int threadsPerWarp = 32);

} // namespace tiny
} // namespace mlir