#include "tiny/Dialect/Tiny/IR/TinyDialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include <sys/_types/_int64_t.h>

#include "mlir/Support/LogicalResult.h"
#include "tiny/Dialect/Tiny/IR/TinyDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "tiny/Dialect/Tiny/IR/TinyTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "tiny/Dialect/Tiny/IR/TinyAttrs.cpp.inc"

#define GET_OP_CLASSES
#include "tiny/Dialect/Tiny/IR/TinyOps.cpp.inc"

namespace mlir::tiny {

void TinyDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "tiny/Dialect/Tiny/IR/TinyTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "tiny/Dialect/Tiny/IR/TinyOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tiny/Dialect/Tiny/IR/TinyAttrs.cpp.inc"
      >();
}

Operation *TinyDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  return ConstantOp::materialize(builder, value, type, loc);
}

// void SliceType::print(AsmPrinter &printer) const {
//   int defaultNum = llvm::maxIntN(32);

//   auto start = getStart();
//   auto end = getEnd();
//   auto stride = getStride();

//   printer << "[" << start;

//   if (end != defaultNum) {
//     printer << "," << end;
//   }

//   if (stride != defaultNum) {
//     printer << "," << stride;
//   }

//   printer << "]";
// }

// Type SliceType::parse(AsmParser &parser) {
//   int defaultNum = llvm::maxIntN(32);
//   int start, end = defaultNum, stride = defaultNum;

//   if (parser.parseLSquare().failed() || parser.parseInteger(start).failed())
//   {
//     return {};
//   }

//   if (parser.parseOptionalComma().succeeded()) {
//     auto endParsed = parser.parseInteger(end);
//   }

//   if (parser.parseOptionalComma().succeeded()) {
//     auto strideParsed = parser.parseInteger(stride);
//   }

//   if (parser.parseRSquare().failed()) {
//     return {};
//   }

//   return parser.getChecked<SliceType>(parser.getContext(), start, end,
//   stride);
// }
} // namespace mlir::tiny