#ifndef ACCL_ATTRS
#define ACCL_ATTRS

include "tiny/Dialect/Tiny/IR/TinyTypes.td"

include "AcclDialect.td"

include "mlir/IR/AttrTypeBase.td"

class Accl_Attr<string name, string attrMnemonic, list<Trait> traits = []>
                : AttrDef<Accl_Dialect, name, traits, "::mlir::Attribute"> {
  let cppNamespace = "::mlir::tiny::accl";
  let description = [{}];
  let mnemonic = "tiny.accl." # attrMnemonic;
}

def ThreadBlockTileAttr : Accl_Attr<"ThreadBlockTile", "thread_block_tile"> {
  let parameters = (ins "unsigned": $N,
                        "unsigned": $M,
                        "unsigned": $K);
  let hasCustomAssemblyFormat = 1;

}

def WarpTileAttr : Accl_Attr<"WarpTile", "warp_tile"> {
  let parameters = (ins "unsigned": $N,
                        "unsigned": $M,
                        "unsigned": $K);

  let hasCustomAssemblyFormat = 1;
}

def ThreadTileAttr : Accl_Attr<"ThreadTile", "thread_tile"> {
  let parameters = (ins "unsigned": $N,
                        "unsigned": $M,
                        "unsigned": $K);
  let hasCustomAssemblyFormat = 1;
}

def CTALayoutAttr : Accl_Attr<"CTALayout", "cta_layout"> {
  let parameters = (ins "ThreadBlockTileAttr":$thread_block_tile,
                        "WarpTileAttr":$warp_tile,
                        "ThreadTileAttr":$thread_tile);
  
  
  let hasCustomAssemblyFormat = 1;
}

def TensorCoreAttr : Accl_Attr<"TensorCore", "tensor_core"> {
  let parameters = (ins "unsigned":$N,
                        "unsigned":$M,
                        "unsigned":$K,
                        "Type":$DtypeIn, 
                        "Type":$DtypeOut);
  let hasCustomAssemblyFormat = 1;
}

class GEMMEncoding<string name, string attrMnemonic, list<Trait> traits = []> : Accl_Attr<name, attrMnemonic, traits> {
  let description = [{
    A GEMM encoding described by Thread-Block Tiles, Warp Tiles, Thread Tiles, and Tensor Cores.
  }];

  code extraGEMMDeclaration = [{
    SmallVector<unsigned> getThreadBlockTileOrder();
    SmallVector<unsigned> getWarpTileOrder();
    SmallVector<unsigned> getThreadTileOrder();
  }];
}

def AmpereGEMMEncodingAttr : GEMMEncoding<"AmpereGEMMEncoding", "ampere_gemm_encoding"> {
  let mnemonic = "ampere_gemm";

  let description = [{
    Encoding that describes valid a GEMM design for the Ampere architecture
  }];

  let parameters = (ins "CTALayoutAttr":$cta_layout,
                        "TensorCoreAttr":$tensor_core);

  let extraClassDeclaration = extraGEMMDeclaration;
  let hasCustomAssemblyFormat = 1;
}

#endif // ACCL_ATTRS