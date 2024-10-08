#ifndef TINY_TYPES
#define TINY_TYPES

include "tiny/Dialect/Tiny/IR/TinyDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/OpBase.td"

/* 
--------------------------------------------------- 
-------------- Tensor & Element Types -------------
--------------------------------------------------- */

/*
Note:
    Seems like the defacto way to represent integer types is through
    signless integers. From what I can gather, it seems that signless
    keeps IR simple and high-level enough with integer bit width.

    My understanding could be completely wrong but it makes sense.
    The only wrench is figuring out what signedness does the int have.
    Seems like there interfaces for figuring out the integer ranges
    of an operations operands at mlir/Interfaces/Utils/InferIntRangeCommon.h

    Can probably use the inferX functions when writing to LLVM.

New Note 10/8/24:
    As it turns out, it seems that my original method of using the interfaces above won't work
    since we still need to know the signedness of the ints and we can't extrapolate that from
    operations which also assume signless integers unless we use the SI or UI types.

    Doing a bit of research, I can see that using the different types isn't standard and the
    standard is to use different typed ops to infer the type (s,u,f)-Op. However, this would
    change the initial specifications of Tiny. In an effort to stay true to the original specs,
    I'm going to use the different types provided instead of signless-integer. If the implementation
    becomes annoying, I'll most likley conform to the standard. I currently cannot see reason not to
    but I'm sure I'll learn soon :).

New New Note 10/8/24:
    I hated it. Going to define new ops that define the types. The thing that TinyGrad has is that it is
    able to change the op based on the type of the Tensor with a ternary operator. The expression of the op
    is defined by the operands which is much easier in Python than in MLIR.

    I can elaborate more on this but essentially TinyGrad can dynamically change the operations rendered through
    operand type checking. We can do that here, but I am unsure if the rest of the dialects used in MLIR would accept
    not using signless-integers since there is only one use of SI in the entire MLIR repo.

    Maybe it isn't as bad as I think it is but re-defining all of the types (SI, UI, I) looked ugly.
*/

def Tiny_Float : AnyTypeOf<[BF16, F16, F32, F64]>;
def Tiny_Int : AnyTypeOf<[I1, I8, I16, I32, I64]>;
def Tiny_Bit : AnyTypeOf<[I1]>;

def Tiny_ElementType : AnyTypeOf<[Tiny_Float, Tiny_Int, Tiny_Bit]>;

def Tiny_ScalarTensor : 0DTensorOf<[Tiny_ElementType]>;
def Tiny_ScalarFloatTensor : 0DTensorOf<[Tiny_Float]>;
def Tiny_ScalarIntTensor : 0DTensorOf<[Tiny_Int]>;
def Tiny_ScalarBitTensor : 0DTensorOf<[Tiny_Bit]>;

def Tiny_Tensor : RankedTensorOf<[Tiny_ElementType]>;
def Tiny_FloatTensor : RankedTensorOf<[Tiny_Float]>;
def Tiny_IntTensor : RankedTensorOf<[Tiny_Int]>;
def Tiny_BitTensor : RankedTensorOf<[Tiny_Bit]>;

def Tiny_MemRef : Non0RankedMemRefOf<[Tiny_ElementType]>;
def Tiny_FloatMemRef : Non0RankedMemRefOf<[Tiny_Float]>;
def Tiny_IntMemRef : Non0RankedMemRefOf<[Tiny_Int]>;
def Tiny_BitMemRef : Non0RankedMemRefOf<[Tiny_Bit]>;

def Tiny_TensorOrMemRef : AnyTypeOf<[Tiny_Tensor, Tiny_MemRef]>;
def Tiny_FloatTensorOrMemRef : AnyTypeOf<[Tiny_FloatTensor, Tiny_FloatMemRef]>;
def Tiny_IntTensorOrMemRef : AnyTypeOf<[Tiny_IntTensor, Tiny_IntMemRef]>;
def Tiny_BitTensorOrMemRef : AnyTypeOf<[Tiny_BitTensor, Tiny_BitMemRef]>;

def Tiny_AnyType : AnyTypeOf<[Tiny_TensorOrMemRef, Tiny_ElementType]>;
def Tiny_AnyFloatType : AnyTypeOf<[Tiny_FloatTensorOrMemRef, Tiny_Float]>;
def Tiny_AnyIntType : AnyTypeOf<[Tiny_IntTensorOrMemRef, Tiny_Int]>;
def Tiny_AnyBitType : AnyTypeOf<[Tiny_BitTensorOrMemRef, Tiny_Bit]>;

/* 
--------------------------------------------------- 
-------------------- Types -----------------------
--------------------------------------------------- */

class Tiny_Type<string name, list<Trait> traits = [], 
                string baseCppClass = "::mlir::Type"> 
    : TypeDef<Tiny_Dialect, name, traits, baseCppClass> {
    let mnemonic = ?;
}

// Note: Syntatic sugar that will get transformed into
// an affine map. Mostly just for ease of use in the IR.
def Tiny_SliceType : Tiny_Type<"Slice"> {  
    let parameters = (ins "int64_t": $start,
                          DefaultValuedParameter<"std::optional<int64_t>", "-1">: $end,
                          DefaultValuedParameter<"std::optional<int64_t>", "1">: $stride);
    let mnemonic = "slice";

    let assemblyFormat =  "`<` $start (`,` $end^ (`,` $stride^)? )? `>`";
}

def Tiny_ShapeType : Tiny_Type<"Shape", [ShapedTypeInterface]> {  
    let parameters = (ins ArrayRefParameter<"int64_t">:$shape, "Type":$elementType);
    let mnemonic = "shape";
    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        class Builder;

        /// Returns if this type is ranked (always true).
        bool hasRank() const { return true; }

        /// Clone this shape type with the given shape and element type. If the
        /// provided shape is `std::nullopt`, the current shape of the type is used.
        ShapeType cloneWith(std::optional<ArrayRef<int64_t>> shape, Type elementType) const;
    }];
}

#endif // TINY_TYPES