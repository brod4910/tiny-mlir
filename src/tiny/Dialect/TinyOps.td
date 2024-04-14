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

class UnaryOp<string mnemonic> : Tiny_Op<mnemonic, [ElementwiseMappable]>;

def NoOp : UnaryOp<"noop"> {

    let description = [{
        A No-Op operation. Undecided but this operation will most likely
        get cannonicalized and removed from the graph.
    }];

    let arguments = (ins AnyTensor:$value);

    let results = (outs AnyTensor:$output);
}

def Exp2 : UnaryOp<"exp2"> {
    let summary = "Two raised by value.";

    let description = [{
        An operation that raises 2 by the value.
    }];

    let arguments = (ins AnyTensor:$value);

    let results = (outs AnyTensor:$output);
}

def Log2 : UnaryOp<"log2"> {
    let summary = "Binary (base-2) logarithm of the value.";

    let description = [{
        An operation that computes the binary (base-2) logarithm of the value.
    }];

    let arguments = (ins AnyTensor:$value);

    let results = (outs AnyTensor:$output);
}

def Cast : UnaryOp<"cast"> {
    let summary = "Cast the value to another type.";

    let description = [{
        An operation that casts the value to another qualified-type.
    }];

    let arguments = (ins AnyTensor:$value);

    let results = (outs AnyTensor:$output);
}

def Sin : UnaryOp<"sin"> {
    let summary = "Sine of the value.";

    let description = [{
        An operation that computes the sine of the value.
    }];

    let arguments = (ins AnyTensor:$value);

    let results = (outs AnyTensor:$output);
}

def Sqrt : UnaryOp<"sqrt"> {
    let summary = "Square-root of the value";

    let description = [{
        An operation that computes the square-root of the value.
    }];

    let arguments = (ins AnyTensor:$value);

    let results = (outs AnyTensor:$output);
}

def Neg : UnaryOp<"neg"> {
    let summary = "Negate the value.";

    let description = [{
        An operation that computes the negation of the value.
    }];

    let arguments = (ins AnyTensor:$value);

    let results = (outs AnyTensor:$output);
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