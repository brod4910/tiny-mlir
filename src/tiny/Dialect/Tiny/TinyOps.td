#ifndef TINY_OPS
#define TINY_OPS

include "TinyDialect.td"
include "TinyTypes.td"

include "mlir/IR/OpBase.td"

class Tiny_Op <string mnemonic, list<Trait> traits = []> : Op<Tiny_Dialect, mnemonic, traits>;

/* 
--------------------------------------------------- 
-------------------- Traits -----------------------
--------------------------------------------------- */


/* 
--------------------------------------------------- 
------------------- UNARY OPS ---------------------
--------------------------------------------------- */

class Tiny_UnaryOp<string mnemonic> : Tiny_Op<mnemonic, [ElementwiseMappable]> {
    let arguments = (ins AnyTensor:$value);

    let results = (outs AnyTensor:$output);

    let assemblyFormat = "$value attr-dict `:` type($value) `->` type($output)";
}

def Tiny_NoOp : Tiny_UnaryOp<"noop"> {
    let summary = "No-operation.";

    let description = [{
        A No-Op operation. Undecided but this operation will most likely
        get cannonicalized and removed from the graph.
    }];
}

def Tiny_Exp2 : Tiny_UnaryOp<"exp2"> {
    let summary = "Two raised by value.";

    let description = [{
        An operation that raises 2 by the value.
    }];
}

def Tiny_Log2 : Tiny_UnaryOp<"log2"> {
    let summary = "Binary (base-2) logarithm of the value.";

    let description = [{
        An operation that computes the binary (base-2) logarithm of the value.
    }];
}

def Tiny_Cast : Tiny_UnaryOp<"cast"> {
    let summary = "Cast the value to another type.";

    let description = [{
        An operation that casts the value to another qualified-type.
    }];
}

def Tiny_Sin : Tiny_UnaryOp<"sin"> {
    let summary = "Sine of the value.";

    let description = [{
        An operation that computes the sine of the value.
    }];
}

def Tiny_Sqrt : Tiny_UnaryOp<"sqrt"> {
    let summary = "Square-root of the value";

    let description = [{
        An operation that computes the square-root of the value.
    }];
}

def Tiny_Neg : Tiny_UnaryOp<"neg"> {
    let summary = "Negate the value.";

    let description = [{
        An operation that computes the negation of the value.
    }];
}

/* 
--------------------------------------------------- 
------------------- REDUCE OPS --------------------
--------------------------------------------------- */

/* 
--------------------------------------------------- 
------------------- BINARY OPS --------------------
--------------------------------------------------- */

/* 
--------------------------------------------------- 
-------------------- LOAD OPS ---------------------
--------------------------------------------------- */

/* 
--------------------------------------------------- 
------------------ TERNARY OPS --------------------
--------------------------------------------------- */


#endif // TINY_OPS