#ifndef TINY_TYPES
#define TINY_TYPES

include "tiny/Dialect/Tiny/IR/TinyDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/OpBase.td"
/* 
--------------------------------------------------- 
-------------------- Types -----------------------
--------------------------------------------------- */

/* 
--------------------------------------------------- 
------------------ Tensor Types -------------------
--------------------------------------------------- */

def Tiny_Float : AnyTypeOf<[BF16, F16, F32, F64]>;

def Tiny_Int : AnyTypeOf<[I1, I8, I16, I32, I64]>;

def Tiny_ElementType : AnyTypeOf<[Tiny_Float, Tiny_Int]>;

def Tiny_Tensor : RankedTensorOf<[Tiny_ElementType]>;
def Tiny_FloatTensor : RankedTensorOf<[Tiny_Float]>;

def Tiny_MemRef : Non0RankedMemRefOf<[Tiny_ElementType]>;
def Tiny_FloatMemRef : Non0RankedMemRefOf<[Tiny_Float]>;

def Tiny_ShapedType : AnyTypeOf<[Tiny_Tensor, Tiny_MemRef]>;
def Tiny_FloatShapedType : AnyTypeOf<[Tiny_FloatTensor, Tiny_FloatMemRef]>;

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