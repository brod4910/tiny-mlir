#ifndef TINY_OPS
#define TINY_OPS


include "tiny/Dialect/Tiny/IR/TinyDialect.td"
include "tiny/Dialect/Tiny/IR/TinyTypes.td"
include "tiny/Dialect/Tiny/IR/TinyAttrs.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"



class Tiny_Op <string mnemonic, list<Trait> traits = []> : Op<Tiny_Dialect, mnemonic, traits>;

/* 
--------------------------------------------------- 
------------------ Constraints --------------------
--------------------------------------------------- */

/* 
--------------------------------------------------- 
-------------------- Traits -----------------------
--------------------------------------------------- */

def ElementwiseBroadcastable : NativeOpTrait<"ElementwiseBroadcastable"> {
  let cppNamespace = "mlir::tiny";
}

/* 
--------------------------------------------------- 
------------------ UTILITY OPS --------------------
--------------------------------------------------- */

def Tiny_SliceOp : Tiny_Op<"slice", [InferTypeOpAdaptor]> {
    let arguments = (ins IndexAttr:$start,
                         DefaultValuedOptionalAttr<IndexAttr, "-1">:$end,
                         DefaultValuedOptionalAttr<IndexAttr, "1">:$stride);
    let results = (outs Tiny_SliceType:$result);

    let assemblyFormat = [{
      `[` $start (`,` $end^ (`,` $stride^)? )? `]` attr-dict `:`  type($result)
    }];
}

def Tiny_ShapeOp : Tiny_Op<"shape", [NoMemoryEffect]> {
    let arguments = (ins I64ArrayAttr: $shape, TypeAttr:$elementType);
    let results = (outs Tiny_ShapeType:$result);

    let assemblyFormat = [{
      $shape `,` $elementType attr-dict `:` type($result)
    }];
}

/* 
--------------------------------------------------- 
-------------------- CONST OP ---------------------
--------------------------------------------------- */

def Tiny_ConstantOp : Tiny_Op<"constant", [ConstantLike, Pure, AllTypesMatch<["value", "result"]>]> {
  let arguments = (ins ElementsAttrInterface: $value);
  let results = (outs Tiny_ShapedType: $result);

  let extraClassDeclaration = [{
    static bool verifyWith(Attribute value, Type type);

    static ConstantOp materialize(OpBuilder &builder, Attribute value, 
                                  Type type, Location loc);
  }];

  let hasFolder = 1;
  let assemblyFormat = "attr-dict $value";
  let hasVerifier = 1;
}

/* 
--------------------------------------------------- 
------------------- UNARY OPS ---------------------
--------------------------------------------------- */

// TODO: Add folders using CommonFolders

class Tiny_UnaryOp<string mnemonic, list<Trait> traits =[]> : Tiny_Op<mnemonic, traits> {
    let arguments = (ins Tiny_ValidType:$value);
    let results = (outs Tiny_ValidType:$result);

    let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";

    // let hasFolder = 1;
}

def Tiny_NoOp : Tiny_UnaryOp<"noop"> {
    let summary = "No-operation.";

    let description = [{
        A No-Op operation.
    }];
}

def Tiny_Exp2Op : Tiny_UnaryOp<"exp2", [ElementwiseMappable, Pure, SameOperandsAndResultShape]> {
    let summary = "Two raised by value.";

    let description = [{
        An operation that raises 2 by the value.
    }];
}

def Tiny_Log2Op : Tiny_UnaryOp<"log2", [ElementwiseMappable, Pure, SameOperandsAndResultShape]> {
    let summary = "Binary (base-2) logarithm of the value.";

    let description = [{
        An operation that computes the binary (base-2) logarithm of the value.
    }];
}

def Tiny_CastOp : Tiny_UnaryOp<"cast", [DeclareOpInterfaceMethods<CastOpInterface>,
                                        Pure,
                                        SameOperandsAndResultShape]> {
    let summary = "Cast the value to another qualified-type.";

    let description = [{
        An operation that casts the value to another qualified-type.
    }];
}

def Tiny_BitcastOp : Tiny_UnaryOp<"Bitcast", [DeclareOpInterfaceMethods<CastOpInterface>,
                                        Pure,
                                        SameOperandsAndResultShape]> {
    let summary = "Cast the value to another type without copying memory.";

    let description = [{
        An operation that casts the value to another qualified-type.
    }];
}

def Tiny_SinOp : Tiny_UnaryOp<"sin", [ElementwiseMappable, Pure, SameOperandsAndResultShape]> {
    let summary = "Sine of the value.";

    let description = [{
        An operation that computes the sine of the value.
    }];
}

def Tiny_SqrtOp : Tiny_UnaryOp<"sqrt", [ElementwiseMappable, Pure, SameOperandsAndResultShape]> {
    let summary = "Square-root of the value";

    let description = [{
        An operation that computes the square-root of the value.
    }];
}

def Tiny_NegOp : Tiny_UnaryOp<"neg", [ElementwiseMappable, Pure, SameOperandsAndResultShape]> {
    let summary = "Negate the value.";

    let description = [{
        An operation that computes the negation of the value.
    }];
}

def Tiny_RecipOp : Tiny_UnaryOp<"recip", [ElementwiseMappable, Pure, SameOperandsAndResultShape]> {
    let summary = "reciprocal of the value.";

    let description = [{
        An operation that computes the reciprocal of the value.
    }];
}

/* 
--------------------------------------------------- 
------------------- REDUCE OPS --------------------
--------------------------------------------------- */

class Tiny_ReduceOp<string mnemonic, list<Trait> traits = []> : Tiny_Op<mnemonic, traits> {
  let arguments = (ins Tiny_ShapedType: $value, 
                   DefaultValuedAttr<I32Attr, "0">: $axis);
  let results = (outs Tiny_ShapedType: $result);

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tiny_MaxOp : Tiny_ReduceOp<"max", [
                                       InferTensorTypeAdaptor,
                                       Pure,
                                      ]> {
  let summary = "Max value along the axis.";

  let description = [{
    An operation that takes the max value along the given axis dimension.
  }];
}

def Tiny_SumOp : Tiny_ReduceOp<"sum", [
                                       InferTensorTypeAdaptor,
                                       Pure,
                                      ]> {
  let summary = "Sum along the axis.";

  let description = [{
    An operation that sums along the given axis dimension.
  }];
}

/* 
--------------------------------------------------- 
------------------- BINARY OPS --------------------
--------------------------------------------------- */

// TODO: Explore elementwise mappable trait for broadcasted
// tensors
//
// TODO: Add folders using CommonFolders

class Tiny_BinaryOp<string mnemonic, list<Trait> traits = []> : Tiny_Op<mnemonic, traits> {
  let arguments = (ins Tiny_ValidType: $lhs, Tiny_ValidType: $rhs);
  let results = (outs Tiny_ValidType: $result);
  
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tiny_AddOp : Tiny_BinaryOp<"add", [
                                       DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>,
                                       Commutative,
                                       ElementwiseMappable,
                                       Pure,
                                       ResultsBroadcastableShape, 
                                       SameOperandsAndResultElementType
                                      ]> {
  let summary = "Add two tensors";
  
  let description = [{
    An operation that adds two tensors.
  }];
}

def Tiny_SubOp : Tiny_BinaryOp<"sub", [
                                       DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>,
                                       ElementwiseMappable,
                                       Pure,
                                       ResultsBroadcastableShape, 
                                       SameOperandsAndResultElementType
                                      ]> {
  let summary = "Subtract two tensors";
  
  let description = [{
    An operation that subtracts one tensor with the other.
  }];
}

def Tiny_MulOp : Tiny_BinaryOp<"mul", [
                                       DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>,
                                       Commutative,
                                       Pure,
                                       ElementwiseBroadcastable,
                                       ResultsBroadcastableShape, 
                                       SameOperandsAndResultElementType
                                      ]> {
  let summary = "Multiply two tensors";
  
  let description = [{
    An operation that multiplies two tensors.
  }];
}

def Tiny_DivOp : Tiny_BinaryOp<"div", [
                                       DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>,
                                       ElementwiseMappable, 
                                       Pure,
                                       ResultsBroadcastableShape, 
                                       SameOperandsAndResultElementType
                                      ]> {
  let summary = "Divide two tensors";
  
  let description = [{
    An operation that divides one tensor with the other.
  }];
}

def Tiny_CmpNeOp : Tiny_BinaryOp<"cmpne", [
                                           DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>,
                                           Commutative,
                                           ElementwiseMappable,
                                           Pure,
                                           ResultsBroadcastableShape, 
                                           SameOperandsAndResultElementType
                                          ]> {
  let summary = "Compares whether the values in the tensor are equal";
  
  let description = [{
    An operation that compares if the values in the two tensors are equal.
  }];
}

def Tiny_CmpLtOp : Tiny_BinaryOp<"cmplt", [
                                           DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>,
                                           ElementwiseMappable,
                                           Pure,
                                           ResultsBroadcastableShape, 
                                           SameOperandsAndResultElementType
                                          ]> {
  let summary = "Compares whether the values in the tensor are less-than";
  
  let description = [{
    An operation that compares if the values in the two tensors are less-than.
  }];
}

def Tiny_MaximumOp : Tiny_BinaryOp<"maximum", [
                                               DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>,
                                               Commutative,
                                               ElementwiseMappable,
                                               Pure, 
                                               ResultsBroadcastableShape, 
                                               SameOperandsAndResultElementType
                                              ]> {
  let summary = "Take the max of the two tensors";
  
  let description = [{
    An operation that takes the max of the two input tensors element-wise
  }];
}

def Tiny_ModOp : Tiny_BinaryOp<"mod", [
                                        DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>,
                                        ElementwiseMappable,
                                        Pure,
                                        ResultsBroadcastableShape, 
                                        SameOperandsAndResultElementType
                                      ]> {
  let summary = "Take the mod of the two tensors";
  
  let description = [{
    An operation that takes the mod of the two input tensors element-wise
  }];
}

def Tiny_XOROp : Tiny_BinaryOp<"xor", [
                                        DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>,
                                        Commutative,
                                        ElementwiseMappable,
                                        Pure, 
                                        ResultsBroadcastableShape, 
                                        SameOperandsAndResultElementType
                                      ]> {
  let summary = "Take the xor of the two tensors";
  
  let description = [{
    An operation that takes the xor of the two input tensors element-wise
  }];
}

def Tiny_ShrOp : Tiny_BinaryOp<"shr", [
                                        DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>,
                                        ElementwiseMappable,
                                        Pure
                                      ]> {
  let arguments = (ins Tiny_ShapedType: $lhs, IndexAttr: $rhs);
  let summary = "Shift LHS right by RHS.";
  
  let description = [{
    An operation that takes the shift-right of LHS by RHS.
  }];

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def Tiny_ShlOp : Tiny_BinaryOp<"shl", [
                                        DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>,
                                        ElementwiseMappable,
                                        Pure
                                      ]> {
  let arguments = (ins Tiny_ShapedType: $lhs, IndexAttr: $rhs);
  let summary = "Shift LHS left by RHS.";
  
  let description = [{
    An operation that takes the shift-right of of LHS by RHS.
  }];

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

/* 
--------------------------------------------------- 
-------------------- LOAD OPS ---------------------
--------------------------------------------------- */

class Tiny_LoadOp<string mnemonic, list<Trait> traits = []> : Tiny_Op<mnemonic, traits>;

// TODO: Pure?
def Tiny_EmptyOp : Tiny_LoadOp<"empty", [InferTensorTypeAdaptor]> {
  let arguments = (ins Tiny_ShapeType: $shape);
  let results = (outs Tiny_ShapedType: $result);

  let summary = "Initializes an empty tensor.";

  let description = [{
    Initializes an empty tensor of the specific shape and type.
  }];


  let assemblyFormat = "$shape  attr-dict `:` functional-type(operands, results)";
}

// TODO: Pure?
def Tiny_CopyOp : Tiny_LoadOp<"copy", [SameOperandsAndResultShape]> {
  let arguments = (ins Tiny_ShapedType: $value);
  let results = (outs Tiny_ShapedType: $result);

  let summary = "Copies a tensor.";

  let description = [{
    Initializes a new tensor by copying another.
  }];

  let assemblyFormat = "$value  attr-dict `:` functional-type(operands, results)";
}

// TODO: Pure?
def Tiny_ContiguousOp : Tiny_LoadOp<"contiguous", [SameOperandsAndResultShape]> {
  let arguments = (ins Tiny_ShapedType: $value);
  let results = (outs Tiny_ShapedType: $result);

  let summary = "Makes a tensor contiguous.";

  let description = [{
    Initializes a new tensor by making it contiguous. This this function will be most useful 
    from a tiny.load or a tiny.view. Otherwise, use copy.
  }];

  let assemblyFormat = "$value  attr-dict `:` functional-type(operands, results)";
}

def Tiny_ViewOp : Tiny_LoadOp<"view", [Pure, InferTensorTypeAdaptor]> {
  let arguments = (ins Tiny_ShapedType: $value, Tiny_ShapeType:$shape);
  let results = (outs Tiny_ShapedType: $result);

  let summary = "Creates a view of a tensor.";

  let description = [{
    Initializes a new tensor which is a view of the operand.
  }];

  let assemblyFormat = "$value `,` $shape attr-dict `:` functional-type(operands, results)";
}

// TODO: Pure?
def Tiny_RandNOp : Tiny_LoadOp<"randn", [SameOperandsAndResultShape]> {
  let arguments = (ins Tiny_ShapeType: $shape);
  let results = (outs Tiny_FloatShapedType: $result);

  let summary = "Makes a random tensor.";

  let description = [{
    Initializes a new tensor with random values between 0-1.
  }];

  let assemblyFormat = "$shape attr-dict `:` functional-type(operands, results)";
}

/* 
--------------------------------------------------- 
------------------- BUFFER OPS --------------------
--------------------------------------------------- */

class Tiny_BufferOp<string mnemonic, list<Trait> traits = []> : Tiny_Op<mnemonic, traits>;

def Tiny_LoadOp : Tiny_BufferOp<"load", [InferTensorTypeAdaptor,
                                         /* MemRefsNormalizable,*/
                                         /*DeclareOpInterfaceMethods<PromotableMemOpInterface>,
                                           DeclareOpInterfaceMethods<DestructurableAccessorOpInterface>*/
                                         ]> {
  let arguments = (ins Arg<Tiny_ShapedType, "source to load from",
                           [MemRead]>: $value,
                       Variadic<Tiny_SliceType>:$slice);
  let results = (outs Tiny_ShapedType: $result);

  let summary = "Load value into memory";

  let description = [{
    Loads a value into a mutable address space.
  }];

  let assemblyFormat = "$value (`[` $slice^ `]`)? attr-dict `:` functional-type(operands, results)";
  // let hasFolder = 1;
}

def Tiny_StoreOp : Tiny_BufferOp<"store", [AttrSizedOperandSegments,
                                           /*DeclareOpInterfaceMethods<PromotableMemOpInterface>,
                                           DeclareOpInterfaceMethods<DestructurableAccessorOpInterface>*/]> {
  let arguments = (ins  Tiny_ShapedType:$src,
                        Variadic<Tiny_SliceType>:$src_slice,
                        Arg<Tiny_ShapedType, "target reference to store to", 
                            [MemWrite]>: $dst,
                        Variadic<Tiny_SliceType>:$dst_slice);
  let results = (outs);

  let summary = "Store src into memory";

  let description = [{
    Stores a value into a mutable address space.
  }];

  // TODO: investigate why Op trait for infering type of $src not working.
  let assemblyFormat = [{
    $src (`[` $src_slice^ `]`)? `,` $dst (`[` $dst_slice^ `]`)? attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
  // let hasFolder = 1;
}

/* 
--------------------------------------------------- 
------------------ TERNARY OPS --------------------
--------------------------------------------------- */

class Tiny_TernaryOp<string mnemonic, list<Trait> traits = []> :  Tiny_Op<mnemonic, traits> {
  let arguments = (ins Tiny_ShapedType: $a, 
                       Tiny_ShapedType: $b, 
                       Tiny_ShapedType: $c);

  let results = (outs Tiny_ShapedType: $result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Tiny_WhereOp : Tiny_TernaryOp<"where", [
                                            InferTensorTypeAdaptor,
                                            ElementwiseMappable,
                                            Pure
                                           ]> {
  // override args since they have more semantic meaning for where ops.
  let arguments = (ins Tiny_ShapedType: $condition, 
                       Tiny_ShapedType: $input, 
                       Tiny_ShapedType: $other);
  let summary = "Produces a tensor of input or other depending on condition";

  let description = [{
    An operation that produces a tensor depending on the condition tensor.
    Taking elements from input or other if true or false.
  }];
}

def Tiny_MulAccOp : Tiny_TernaryOp<"mulacc", [
                                              InferTensorTypeAdaptor,
                                              ElementwiseMappable,
                                              Pure
                                             ]> {
  let summary = "Computes a multiply-accumulate operation in the form of: (a * b) + c";

  let description = [{
    An operation that computes the multiply-accumulate operation.
  }];
}

/* 
--------------------------------------------------- 
------------------ FUNCTION OPS -------------------
--------------------------------------------------- 

            COPIED FROM THE MLIR REPO

Note:
  Mulling over this section for now. I don't think Tiny needs or wants special
  Function semantics but that could change in the future. I found that modifying or
  creating new function semantics to be rather difficult at this point in time (July 9 2024).

  This section will just get converted to the Func dialect but may change in the future.
  Triton modifies the func operators for inlining but i belive function inlining has improved
  through the use of the Inliner pass located at mlir/Transforms/Passes.td not tested at this time
  and this is just an assumption to be proven correct or incorrect later.
*/


def Tiny_CallOp : Tiny_Op<"call",
    [CallOpInterface, MemRefsNormalizable,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "call operation";
  let description = [{
    The `tiny.call` operation represents a direct call to a function that is
    within the same symbol scope as the call. The operands and result types of
    the call must match the specified function type. The callee is encoded as a
    symbol reference attribute named "callee".

    Example:

    ```mlir
    %2 = tiny.call @my_add(%0, %1) : (f32, f32) -> f32
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, SymbolRefAttr::get(callee), results, operands);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, StringAttr::get($_builder.getContext(), callee),
            results, operands);
    }]>];

  let extraClassDeclaration = [{
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def Tiny_FuncOp : Tiny_Op<"func", [
  AffineScope, AutomaticAllocationScope,
  FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface
]> {
  let summary = "An operation with a name containing a single `SSACFG` region";
  let description = [{
    Operations within the function cannot implicitly capture values defined
    outside of the function, i.e. Functions are `IsolatedFromAbove`. All
    external references must use function arguments or attributes that establish
    a symbolic connection (e.g. symbols referenced by name via a string
    attribute like SymbolRefAttr). An external function declaration (used when
    referring to a function declared in some other module) has no body. While
    the MLIR textual form provides a nice inline syntax for function arguments,
    they are internally represented as “block arguments” to the first block in
    the region.

    Only dialect attribute names may be specified in the attribute dictionaries
    for function arguments, results, or the function itself.

    Example:

    ```mlir
    // External function definitions.
    func.func private @abort()
    func.func private @scribble(i32, i64, memref<? x 128 x f32, #layout_map0>) -> f64

    // A function that returns its argument twice:
    func.func @count(%x: i64) -> (i64, i64)
      attributes {fruit: "banana"} {
      return %x, %x: i64, i64
    }

    // A function with an argument attribute
    func.func private @example_fn_arg(%x: i32 {swift.self = unit})

    // A function with a result attribute
    func.func private @example_fn_result() -> (f64 {dialectName.attrName = 0 : i64})

    // A function with an attribute
    func.func private @example_fn_attr() attributes {dialectName.attrName = false}
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];
  let extraClassDeclaration = [{
    static FuncOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs = {});
    static FuncOp create(Location location, StringRef name, FunctionType type,
                         Operation::dialect_attr_range attrs);
    static FuncOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs,
                         ArrayRef<DictionaryAttr> argAttrs);

    /// Create a deep copy of this function and all of its blocks, remapping any
    /// operands that use values outside of the function using the map that is
    /// provided (leaving them alone if no entry is present). If the mapper
    /// contains entries for function arguments, these arguments are not
    /// included in the new function. Replaces references to cloned sub-values
    /// with the corresponding value that is copied, and adds those mappings to
    /// the mapper.
    FuncOp clone(IRMapping &mapper);
    FuncOp clone();

    /// Clone the internal blocks and attributes from this function into dest.
    /// Any cloned blocks are appended to the back of dest. This function
    /// asserts that the attributes of the current function and dest are
    /// compatible.
    void cloneInto(FuncOp dest, IRMapping &mapper);

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "func"; }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }
  }];
  let hasCustomAssemblyFormat = 1;
}

def Tiny_ReturnOp : Tiny_Op<"return", [Pure, HasParent<"FuncOp">,
                                MemRefsNormalizable, ReturnLike, Terminator]> {
  let summary = "Function return operation";
  let description = [{
    The `func.return` operation represents a return operation within a function.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the function
    that contains the operation.

    Example:

    ```mlir
    tiny.func @foo() : (i32, f8) {
      ...
      tiny.return %0, %1 : i32, f8
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

#endif // TINY_OPS