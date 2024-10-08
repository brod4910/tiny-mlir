#include "tiny/Dialect/Tiny/IR/TinyDialect.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
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
------------------- TINY TRAITS --------------------
--------------------------------------------------- */

/* -------- ElementwiseBroadcastable Trait -------- */

bool verifyElementwise(Operation *op) {
  auto isRankedTensorType = [](Type type) {
    return llvm::isa<RankedTensorType>(type);
  };

  auto resultMappableTypes = llvm::to_vector<1>(
      llvm::make_filter_range(op->getResultTypes(), isRankedTensorType));
  auto operandMappableTypes = llvm::to_vector<2>(
      llvm::make_filter_range(op->getOperandTypes(), isRankedTensorType));

  // If the op only has scalar operand/result types, then we have nothing to
  // check.
  if (resultMappableTypes.empty() && operandMappableTypes.empty())
    return true;

  if (!resultMappableTypes.empty() && operandMappableTypes.empty())
    return false;

  assert(!operandMappableTypes.empty());

  if (resultMappableTypes.empty())
    return false;

  if (resultMappableTypes.size() != op->getNumResults())
    return false;

  SmallVector<Type, 4> types = llvm::to_vector<2>(
      llvm::concat<Type>(operandMappableTypes, resultMappableTypes));
  TypeID expectedBaseTy = types.front().getTypeID();
  if (!llvm::all_of(types,
                    [&](Type t) { return t.getTypeID() == expectedBaseTy; }) ||
      failed(verifyCompatibleShapes(types))) {
    return false;
  }

  return true;
}

bool hasElementwiseBroadcastableTrait(Operation *op) {
  return op->hasTrait<ElementwiseBroadcastable>();
}

bool isElementwiseBroadcastableOpOnRankedTensors(Operation *op) {
  if (!hasElementwiseBroadcastableTrait(op))
    return false;

  return llvm::all_of(op->getOperandTypes(),
                      [](Type type) { return isa<RankedTensorType>(type); });
}

bool isElementwiseOpOnRankedTensors(Operation *op) {
  if (!OpTrait::hasElementwiseMappableTraits(op))
    return false;

  return llvm::all_of(op->getOperandTypes(),
                      [](Type type) { return isa<RankedTensorType>(type); });
}

/* ---------------- Reducer Trait ----------------- */

bool hasReducerTrait(Operation *op) { return op->hasTrait<Reducer>(); }

bool isReducerOpOnRankedTensors(Operation *op) {
  if (!hasReducerTrait(op))
    return false;

  return llvm::all_of(op->getOperandTypes(),
                      [](Type type) { return isa<RankedTensorType>(type); });
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
  AffineExpr bounds;

  // Slice will always be equal to the length of the tensor being operated on
  // through shape inference on Ops that use slicing mechanics.
  for (const auto &en : llvm::enumerate(slices)) {
    auto dim = en.index();
    auto slice = en.value();

    auto d = getAffineDimExpr(dim, context);
    auto bounds_d = getAffineDimExpr(dim, context);

    AffineExpr start = getAffineConstantExpr(slice.getStart(), context);
    AffineExpr end = getAffineConstantExpr(*slice.getEnd(), context);
    AffineExpr stride = getAffineConstantExpr(*slice.getStride(), context);

    expr = expr + (start + d * stride);

    bounds = bounds + ((end - start).floorDiv(stride));
  }

  return AffineMap::get(slices.size(), 0, {expr, bounds}, context);
}

/* -------------- Sliced Layout Attr -------------- */
AffineMap SlicedLayoutAttr::getAffineMap() const {
  return makeSlicedLayoutMap(getSlices(), getContext());
}

} // namespace mlir::tiny