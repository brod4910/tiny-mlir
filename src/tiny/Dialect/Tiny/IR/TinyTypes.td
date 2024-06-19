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

def Tiny_Tensor : RankedTensorOf<[Tiny_Int, Tiny_Float]>;

class Tiny_Type<string name, list<Trait> traits = [], 
                string baseCppClass = "::mlir::Type"> 
    : TypeDef<Tiny_Dialect, name, traits, baseCppClass> {
    let mnemonic = ?;
}

// Note: Syntatic sugar that will get transformed into
// an affine map. Mostly just for ease of use in the IR.
def Tiny_SliceType : Tiny_Type<"Slice"> {  
    let parameters = (ins DefaultValuedParameter<"std::optional<int64_t>", "-1">: $start,
                        DefaultValuedParameter<"std::optional<int64_t>", "-1">: $end,
                        DefaultValuedParameter<"std::optional<int64_t>", "1">: $stride);
    let mnemonic = "slice";

    // Note: In essence:
    // if (start) {
    //    print or parse
    //    if (end) {
    //      print or parse
    //      if (stride) {
    //        print or parse
    //      }
    //    }
    // }
    let assemblyFormat =  "`<` ( $start^ (`,` $end^ (`,` $stride^)? )? )? `>`";

    let extraClassDeclaration = [{
      bool startIsDefault() {
        return *getStart() == -1;
      }

      bool endIsDefault() {
        return *getEnd() == -1;
      }

      bool strideIsDefault() {
        return *getStride() == 1;
      }

      static bool isDefaultSlice(mlir::tiny::SliceType slice) {
        return slice == mlir::tiny::SliceType{};
      }
    }];
}

#endif // TINY_TYPES