#ifndef TINY_TYPES
#define TINY_TYPES

include "TinyDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

/* 
--------------------------------------------------- 
-------------------- Types -----------------------
--------------------------------------------------- */

class Tiny_Type<string name, list<Trait> traits = [], 
                string baseCppClass = "::mlir::Type"> 
    : TypeDef<Tiny_Dialect, name, traits, baseCppClass> {
    let mnemonic = ?;
}

class Tiny_Tensor : Tiny_Type<"TinyTensor", [ShapedTypeInterface], "TensorType"> {
    let summary = "A tensor with int or float-like values.";

    let description = [{
        A type for tensor-like objects.
    }];

    let parameters = (ins
     ArrayRefParameter<"int64_t">:$shape,
     "Type":$elementType,
     "Attribute":$encoding
    );
}



#endif // TINY_TYPES