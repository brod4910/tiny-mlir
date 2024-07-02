#pragma once

#include <memory>
#include <string>

namespace mlir {
class ModuleOp;
template <typename T> class OperationPass;

namespace tiny {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTinyToLLVMPass();

} // namespace tiny
} // namespace mlir