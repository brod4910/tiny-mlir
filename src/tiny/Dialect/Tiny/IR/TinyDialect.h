#pragma once
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/CastInterfaces.h"

#include "tiny/Dialect/Tiny/IR/TinyDialect.h.inc"

#define GET_OP_CLASSES
#include "tiny/Dialect/Tiny/IR/TinyOps.h.inc"