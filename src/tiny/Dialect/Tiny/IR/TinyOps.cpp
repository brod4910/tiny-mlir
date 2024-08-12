#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Casting.h"

#include "tiny/Dialect/Tiny/IR/TinyDialect.h"

namespace mlir::tiny {
/*
---------------------------------------------------
------------------ UTILITY OPS --------------------
--------------------------------------------------- */

/* ------------------ Slice Op ------------------- */

LogicalResult
SliceShapeInference(RankedTensorType valueType, ValueRange slices,
                    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes,
                    Location loc, StringLiteral opName) {

  auto valueShape = valueType.getShape();
  SmallVector<int64_t> resultShape;

  for (auto [size, slice] : llvm::zip_longest(valueShape, slices)) {
    if (!size.has_value() && slice.has_value()) {
      return emitError(loc, opName) << ": shape inference failed. Slice length "
                                       "is greater than tensor shape.";
    }

    SliceType dimSlice;

    if (!slice.has_value()) {
      dimSlice = SliceType::get(valueType.getContext(), 0, *size, 1);
    } else {
      dimSlice = dyn_cast<SliceType>(slice->getType());
    }

    auto start = dimSlice.getStart();
    auto end = dimSlice.getEnd();
    auto stride = dimSlice.getStride();

    if (*size <= start) {
      return emitError(loc, opName)
             << ": shape inference failed. Start is " << start
             << " which is greater than or equal to size which is " << *size;
    } else if (*end > size) {
      return emitError(loc, opName)
             << ": shape inference failed. Size is " << *size
             << " which is greater than end which is " << *end;
    }

    if (*end > 0) {
      auto quotient = (*end - start) / *stride;
      auto remainder = (*end - start) % *stride;

      auto resultSize = quotient + remainder;

      resultShape.push_back(resultSize);
    }
  }

  inferredReturnShapes.emplace_back(resultShape, valueType.getElementType());

  return success();
}

LogicalResult SliceOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    SliceOpAdaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto start = adaptor.getStart();
  auto end = adaptor.getEnd();
  auto stride = adaptor.getStride();

  inferredReturnTypes.push_back(SliceType::get(context, start.getSExtValue(),
                                               end.getSExtValue(),
                                               stride.getSExtValue()));
  return success();
}

/* ------------------ Shape Op ------------------- */
void ShapeOp::build(OpBuilder &builder, OperationState &ods,
                    llvm::ArrayRef<int64_t> shape, Type elementType) {
  // possible to create shape from array ref?
}
/*
---------------------------------------------------
------------------- CONSTANT OP -------------------
--------------------------------------------------- */
bool ConstantOp::verifyWith(Attribute value, Type type) {
  auto rankedType = llvm::dyn_cast<RankedTensorType>(type);

  if (!rankedType) {
    return false;
  }

  if (llvm::isa<IntegerType>(rankedType) &&
      !llvm::cast<IntegerType>(type).isSignless()) {
    return false;
  }

  if (!llvm::isa<DenseIntOrFPElementsAttr>(value)) {
    return false;
  }

  return true;
}

ConstantOp ConstantOp::materialize(OpBuilder &builder, Attribute value,
                                   Type type, Location loc) {
  if (verifyWith(value, type)) {
    return builder.create<ConstantOp>(loc, type,
                                      llvm::cast<ElementsAttr>(value));
  }

  return nullptr;
}

LogicalResult ConstantOp::verify() {
  auto type = llvm::dyn_cast<RankedTensorType>(getType());

  if (!type) {
    return emitOpError("value must be ranked tensor.");
  }

  if (llvm::isa<IntegerType>(type) &&
      !llvm::cast<IntegerType>(type).isSignless()) {
    return emitOpError("integer return type must be signless.");
  }

  if (!llvm::isa<DenseIntOrFPElementsAttr>(getValue())) {
    return emitOpError(
        "value must be integer or floating-point elements attributes.");
  }

  return success();
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

/*
---------------------------------------------------
------------------- UNARY OPS ---------------------
--------------------------------------------------- */

/* ------------------ Cast Op ------------------- */

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    return false;
  }

  auto input = llvm::dyn_cast<TensorType>(inputs.front());
  auto output = llvm::dyn_cast<TensorType>(outputs.front());

  if (!input || !output) {
    return false;
  }

  // The shape is required to match if both types are ranked.
  return succeeded(verifyCompatibleShape(input, output));
}

/* ------------------ Cast Op ------------------- */

bool BitcastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    return false;
  }

  auto input = llvm::dyn_cast<TensorType>(inputs.front());
  auto output = llvm::dyn_cast<TensorType>(outputs.front());

  if (!input || !output) {
    return false;
  }

  // The shape is required to match if both types are ranked.
  return succeeded(verifyCompatibleShape(input, output));
}

/* ------------------ NoOp Op ------------------- */

// OpFoldResult NoOp::fold(FoldAdaptor adaptor) { return getValue(); }

/*
---------------------------------------------------
------------------- REDUCE OPS --------------------
--------------------------------------------------- */

LogicalResult ReduceOpShapeInference(
    RankedTensorType value, int32_t axis,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {

  auto rank = value.getRank();

  // Bounds check for -/+ axis
  if (rank + axis < 0 || axis >= rank) {
    return failure();
  }

  auto valueShape = value.getShape();

  // Get real axis. Rank + (-axis) = real axis
  axis = axis >= 0 ? axis : rank + axis;

  SmallVector<int64_t> resultShape;

  for (int i = 0; i < valueShape.size(); ++i) {
    if (i != axis) {
      resultShape.push_back(valueShape[i]);
    }
  }

  inferredReturnShapes.emplace_back(ArrayRef(resultShape),
                                    value.getElementType());

  return success();
}

/* ------------------ Max Op ------------------- */

LogicalResult MaxOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    MaxOpAdaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  auto value = dyn_cast<RankedTensorType>(adaptor.getValue().getType());
  auto axis = adaptor.getAxis();

  return ReduceOpShapeInference(value, axis, inferredReturnShapes);
}

/* ------------------ Sum Op ------------------- */

LogicalResult SumOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    SumOpAdaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  auto value = dyn_cast<RankedTensorType>(adaptor.getValue().getType());
  auto axis = adaptor.getAxis();

  return ReduceOpShapeInference(value, axis, inferredReturnShapes);
}

/*
---------------------------------------------------
------------------- BINARY OPS --------------------
--------------------------------------------------- */

LogicalResult BinaryOpShapeInference(
    ValueShapeRange operands,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes, Location loc) {
  auto lhs = operands.getValueAsShape(0);
  auto rhs = operands.getValueAsShape(1);

  auto lhsShape = ShapedTypeComponents(lhs);
  auto rhsShape = ShapedTypeComponents(rhs);

  SmallVector<int64_t> resultShape;

  if (!OpTrait::util::getBroadcastedShape(lhsShape.getDims(),
                                          rhsShape.getDims(), resultShape)) {
    return emitError(loc, "binary op shapes not broadcastable.");
  }

  inferredReturnShapes.emplace_back(ArrayRef(resultShape),
                                    lhs.getElementType());

  return success();
}

/* ------------------ Add Op ---------------------- */

LogicalResult AddOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  return BinaryOpShapeInference(operands, inferredReturnShapes, *location);
}

/* ------------------ Sub Op ---------------------- */

LogicalResult SubOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  return BinaryOpShapeInference(operands, inferredReturnShapes, *location);
}

/* ------------------ Mul Op ---------------------- */

LogicalResult MulOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  return BinaryOpShapeInference(operands, inferredReturnShapes, *location);
}

/* ------------------ Div Op ---------------------- */

LogicalResult DivOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  return BinaryOpShapeInference(operands, inferredReturnShapes, *location);
}

/* ----------------- CMPNe Op --------------------- */

LogicalResult CmpNeOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  return BinaryOpShapeInference(operands, inferredReturnShapes, *location);
}

/* ----------------- CMPLT Op --------------------- */

LogicalResult CmpLtOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  return BinaryOpShapeInference(operands, inferredReturnShapes, *location);
}

/* ------------------ Maximum Op ---------------------- */

LogicalResult MaximumOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  return BinaryOpShapeInference(operands, inferredReturnShapes, *location);
}

/* --------------------- Mod Op ---------------------- */

LogicalResult ModOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  return BinaryOpShapeInference(operands, inferredReturnShapes, *location);
}

/* --------------------- XOR Op ---------------------- */

LogicalResult XOROp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  return BinaryOpShapeInference(operands, inferredReturnShapes, *location);
}

/* --------------------- Shl Op ---------------------- */

LogicalResult ShlOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  // TODO: Check if this fails for any cases since operand 1 is IndexAttr
  return BinaryOpShapeInference(operands, inferredReturnShapes, *location);
}

/* --------------------- Shr Op ---------------------- */

LogicalResult ShrOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  // TODO: Check if this fails for any cases since operand 1 is IndexAttr
  return BinaryOpShapeInference(operands, inferredReturnShapes, *location);
}

/*
---------------------------------------------------
-------------------- LOAD OPS ---------------------
--------------------------------------------------- */

LogicalResult
ViewShapeInference(RankedTensorType valueType, ShapeType shapeType,
                   SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes,
                   Location loc, StringLiteral opName) {
  int64_t tensor_numel = valueType.getNumElements();
  int64_t shape_numel = shapeType.getNumElements();

  if (tensor_numel != shape_numel) {
    return emitError(loc, "Tensor and Shape numel not equal: ")
           << tensor_numel << " != " << shape_numel;
  }

  SmallVector<int64_t> resultShape{shapeType.getShape().begin(),
                                   shapeType.getShape().end()};
  inferredReturnShapes.emplace_back(resultShape, shapeType.getElementType());
  return success();
}

/* ------------------ Empty Op -------------------- */

LogicalResult EmptyOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    EmptyOpAdaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  auto shape = dyn_cast<ShapeType>(adaptor.getShape().getType());
  inferredReturnShapes.emplace_back(shape.getShape(), shape.getElementType());
  return success();
}

/* ------------------- View Op -------------------- */

LogicalResult ViewOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ViewOpAdaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  auto valueType = dyn_cast<RankedTensorType>(adaptor.getValue().getType());
  auto shapeType = dyn_cast<ShapeType>(adaptor.getShape().getType());
  return ViewShapeInference(valueType, shapeType, inferredReturnShapes,
                            *location, getOperationName());
}

/*
---------------------------------------------------
------------------- BUFFER OPS --------------------
--------------------------------------------------- */

/* ------------------- Load Op -------------------- */

LogicalResult LoadOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    LoadOpAdaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  auto valueType = dyn_cast<RankedTensorType>(adaptor.getValue().getType());
  auto slices = adaptor.getSlice();

  return SliceShapeInference(valueType, slices, inferredReturnShapes, *location,
                             getOperationName());
}

/* ----------------- Store Op ------------------- */

LogicalResult StoreOp::verify() {
  SmallVector<ShapedTypeComponents> slicedShapes;

  auto srcType = dyn_cast<RankedTensorType>(getSrc().getType());
  auto srcSlices = getSrcSlice();

  auto srcFailed = SliceShapeInference(srcType, srcSlices, slicedShapes,
                                       getLoc(), getOperationName());

  if (srcFailed.failed()) {
    return srcFailed;
  }

  auto dstType = dyn_cast<RankedTensorType>(getDst().getType());
  auto dstSlices = getDstSlice();

  auto dstFailed = SliceShapeInference(dstType, dstSlices, slicedShapes,
                                       getLoc(), getOperationName());

  if (dstFailed.failed()) {
    return dstFailed;
  }

  SmallVector<int64_t> resultShape;

  if (!OpTrait::util::getBroadcastedShape(
          slicedShapes[0].getDims(), slicedShapes[1].getDims(), resultShape)) {
    return emitOpError("src and dst shape aren't broadcastable with shapes: ")
           << slicedShapes[0].getDims() << " and " << slicedShapes[1].getDims();
  }

  return success();
}

/*
---------------------------------------------------
------------------ TERNARY OPS --------------------
--------------------------------------------------- */

/* ------------------ Where Op ------------------- */

LogicalResult WhereOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    WhereOpAdaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  auto conditionType =
      dyn_cast<RankedTensorType>(adaptor.getCondition().getType());
  auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  auto otherType = dyn_cast<RankedTensorType>(adaptor.getOther().getType());

  auto shapes = ArrayRef({SmallVector<int64_t>(conditionType.getShape()),
                          SmallVector<int64_t>(inputType.getShape()),
                          SmallVector<int64_t>(otherType.getShape())});

  if (inputType.getElementType() != otherType.getElementType()) {
    return failure();
  }

  if (!OpTrait::util::staticallyKnownBroadcastable(shapes)) {
    return failure();
  }

  SmallVector<int64_t> resultShape;
  OpTrait::util::getBroadcastedShape(inputType.getShape(), otherType.getShape(),
                                     resultShape);
  inferredReturnShapes.emplace_back(resultShape, inputType.getElementType());

  return success();
}

/* ------------------ MulAcc Op ------------------- */

LogicalResult MulAccOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    MulAccOpAdaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  auto aType = dyn_cast<RankedTensorType>(adaptor.getA().getType());
  auto bType = dyn_cast<RankedTensorType>(adaptor.getB().getType());
  auto cType = dyn_cast<RankedTensorType>(adaptor.getC().getType());

  auto shapes = ArrayRef({SmallVector<int64_t>(aType.getShape()),
                          SmallVector<int64_t>(bType.getShape()),
                          SmallVector<int64_t>(cType.getShape())});

  if (aType.getElementType() != bType.getElementType() ||
      aType.getElementType() != cType.getElementType()) {
    return failure();
  }

  if (!OpTrait::util::staticallyKnownBroadcastable(shapes)) {
    return failure();
  }

  SmallVector<int64_t> resultShape;
  OpTrait::util::getBroadcastedShape(bType.getShape(), cType.getShape(),
                                     resultShape);
  inferredReturnShapes.emplace_back(resultShape, bType.getElementType());

  return success();
}

/*
---------------------------------------------------
------------------ FUNCTION OPS -------------------
---------------------------------------------------

             COPIED FROM THE MLIR REPO
*/

/* ------------------ FuncOp ------------------- */

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  function_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, /*resultAttrs=*/std::nullopt,
      getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(OpAsmPrinter &printer) {
  function_interface_impl::printFunctionOp(
      printer, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

/* ------------------ CallOp ------------------- */

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }

  return success();
}

/* ------------------ ReturnOp ------------------- */

LogicalResult ReturnOp::verify() {
  auto function = cast<FuncOp>((*this)->getParentOp());

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing function (@"
           << function.getName() << ") returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i).getType() != results[i])
      return emitError() << "type of return operand " << i << " ("
                         << getOperand(i).getType()
                         << ") doesn't match function result type ("
                         << results[i] << ")"
                         << " in function @" << function.getName();

  return success();
}

} // namespace mlir::tiny