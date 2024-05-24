#ifndef ACCL_ATTRS
#define ACCL_ATTRS

include "tiny/Dialect/Tiny/IR/TinyTypes.td"

include "AcclDialect.td"

include "mlir/IR/AttrTypeBase.td"

class Accl_Attr<string name, list<Trait> traits = []>
                : AttrDef<Accl_Dialect, name, traits> {
  let cppNamespace = "::mlir::tiny::accl";
}

class Dim3Attr<string name, string attrMnemonic, list<Trait> traits = []> 
                         : Accl_Attr<name, traits> {
  let mnemonic = attrMnemonic;

  let parameters = (ins "unsigned": $M,
                        "unsigned": $N,
                        "unsigned": $K);

  let assemblyFormat = "`<` struct(params) `>`";  
}

def ThreadBlockTileAttr : Dim3Attr<"ThreadBlockTile", "thread_block_tile">;

def WarpTileAttr : Dim3Attr<"WarpTile", "warp_tile">;

def ThreadTileAttr : Dim3Attr<"ThreadTile", "thread_tile">;

def TensorCoreAttr : Dim3Attr<"TensorCore", "tensor_core"> {
  let parameters = (ins "unsigned":$M,
                        "unsigned":$N,
                        "unsigned":$K,
                        "Type":$DtypeIn, 
                        "Type":$DtypeOut);
}

def CTALayoutAttr : Accl_Attr<"CTALayout"> {
  let mnemonic = "cta_layout";

  let parameters = (ins "ThreadBlockTileAttr":$threadBlockTile,
                        "WarpTileAttr":$warpTile,
                        "ThreadTileAttr":$threadTile);

  let builders = [
    AttrBuilder<(ins "ArrayRef<unsigned>":$threadBlockTile,
                     "ArrayRef<unsigned>":$warpTile,
                     "ArrayRef<unsigned>":$threadTile), [{
        auto threadBlockTileAttr = ThreadBlockTileAttr::get(context, threadBlockTile[0], threadBlockTile[1], threadBlockTile[2]);
        auto warpTileAttr = WarpTileAttr::get(context, warpTile[0], warpTile[1], warpTile[2]);
        auto threadTileAttr = ThreadTileAttr::get(context, threadTile[0], threadTile[1], threadTile[2]);

        return $_get(context, threadBlockTileAttr, warpTileAttr, threadTileAttr);
    }]>
  ];

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

class GEMMEncoding<string name, string attrMnemonic, list<Trait> traits = []> : Accl_Attr<name, traits> {
  let mnemonic = attrMnemonic;
  
  let description = [{
    A GEMM encoding described by Thread-Block Tiles, Warp Tiles, Thread Tiles, and Tensor Cores.
  }];

}

#endif // ACCL_ATTRS