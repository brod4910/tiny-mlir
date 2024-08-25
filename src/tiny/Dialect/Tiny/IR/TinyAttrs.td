#ifndef TINY_ATTRS
#define TINY_ATTRS

include "tiny/Dialect/Tiny/IR/TinyDialect.td"
include "tiny/Dialect/Tiny/IR/TinyTypes.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/AttrTypeBase.td"

class Tiny_Attr<string name, string attrMnemonic, list<Trait> traits = [], string baseCppClass = "::mlir::Attribute">
                : AttrDef<Tiny_Dialect, name, traits, baseCppClass> {
  let mnemonic = ?;
  let cppNamespace = "::mlir::tiny";
  let attrName = "tiny." # attrMnemonic;
}

def Tiny_SlicedLayoutAttr : Tiny_Attr<"SlicedLayout", "sliced_layout", [DeclareAttrInterfaceMethods<MemRefLayoutAttrInterface>]> {
  let summary = "An attribute representing a sliced layout of a shaped type.";
  let description = [{
    
  }];

  let parameters = (ins
    ArrayRefParameter<"SliceType", "array of slices">:$slices
  );
}

// MFLAGS = ('nsz', 'arcp', 'contract', 'afn', 'reassoc') # All from fast math,
// but nnan and ninf
// LLVM::FastmathFlags FMF{LLVM::FastmathFlags::nsz | LLVM::FastmathFlags::arcp |
//                         LLVM::FastmathFlags::contract |
//                         LLVM::FastmathFlags::afn |
//                         LLVM::FastmathFlags::reassoc};


def TFMFnsz : I32BitEnumAttrCaseBit<"nsz", 2>;
def TFMFarcp : I32BitEnumAttrCaseBit<"arcp", 3>;
def TFMFcontract : I32BitEnumAttrCaseBit<"contract", 4>;
def TFMFafn : I32BitEnumAttrCaseBit<"afn", 5>;
def TFMFreassoc : I32BitEnumAttrCaseBit<"reassoc", 6>;
def TFMFfast : I32BitEnumAttrCaseGroup<"fast", [TFMFnsz, TFMFarcp, TFMFcontract, TFMFafn, TFMFreassoc]>;

def FastmathFlags : I32BitEnumAttr<"FastmathFlags",
                                   "Tiny Fast Math Flags",
                                   [TFMFnsz, TFMFarcp, TFMFcontract, TFMFafn, TFMFreassoc, TFMFfast]> {
  let separator = ",";
  let cppNamespace = "::mlir::tiny";
  let genSpecializedAttr = 0;
  let printBitEnumPrimaryGroups = 1;
}

def Tiny_FastmathFlagsAttr : EnumAttr<Tiny_Dialect, FastmathFlags, "fastmath"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // TINY_ATTRS