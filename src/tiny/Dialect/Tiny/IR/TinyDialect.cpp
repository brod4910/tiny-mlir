#include "tiny/Dialect/Tiny/IR/TinyDialect.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

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

/*
---------------------------------------------------
------------------- TINY TYPES --------------------
--------------------------------------------------- */

/* ------------------ Shape Type ------------------ */

ShapeType ShapeType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                               Type elementType) const {
  return ShapeType::get(elementType.getContext(), *shape, elementType);
}

Type ShapeType::parse(::mlir::AsmParser &parser) {
  llvm::SmallVector<int64_t> dimensions;
  Type type;

  if (parser.parseLess().failed() ||
      parser.parseDimensionList(dimensions).failed() ||
      parser.parseType(type).failed()) {
    return {};
  }

  return ShapeType::get(parser.getContext(), ArrayRef(dimensions), type);
}

void ShapeType::print(AsmPrinter &printer) const {
  printer << "<";
  printer.printDimensionList(getShape());
  printer << "x" << getElementType() << ">";
}

/*
---------------------------------------------------
----------------- TINY ATTRIBUTES -----------------
--------------------------------------------------- */
AffineMap makeSlicedLayoutMap(ArrayRef<SliceType> slices,
                              MLIRContext *context) {
  AffineExpr expr;

  for (const auto &en : llvm::enumerate(slices)) {
    auto dim = en.index();
    auto slice = en.value();

    auto d = getAffineDimExpr(dim, context);
    AffineExpr mult;
  }
}

/* -------------- Sliced Layout Attr -------------- */
AffineMap SlicedLayoutAttr::getAffineMap() const { return {}; }

LogicalResult SlicedLayoutAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<SliceType> slices) {
  return success();
}

} // namespace mlir::tiny