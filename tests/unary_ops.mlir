// RUN: tiny-opt %s --tiny-erase-noop --tiny-cast-noop | FileCheck %s
tiny.func @simple_sin() -> tensor<3xf32> {
    %0 = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %1 = tiny.sin %0 : tensor<3xf32> -> tensor<3xf32>
    
    tiny.return %1 : tensor<3xf32>
}

tiny.func @simple_sqrt() -> tensor<3xf32> {
    %0 = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %1 = tiny.sqrt %0 : tensor<3xf32> -> tensor<3xf32>
    
    tiny.return %1 : tensor<3xf32>
}

tiny.func @simple_log2() -> tensor<3xf32> {
    %0 = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %1 = tiny.log2 %0 : tensor<3xf32> -> tensor<3xf32>
    
    tiny.return %1 : tensor<3xf32>
}

tiny.func @simple_neg() -> tensor<3xf32> {
    %0 = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %1 = tiny.neg %0 : tensor<3xf32> -> tensor<3xf32>
    
    tiny.return %1 : tensor<3xf32>
}

tiny.func @simple_exp2() -> tensor<3xf32> {
    %0 = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %1 = tiny.exp2 %0 : tensor<3xf32> -> tensor<3xf32>
    
    tiny.return %1 : tensor<3xf32>
}

tiny.func @simple_noop() -> tensor<3xf32> {
    %0 = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %1 = tiny.noop %0 : tensor<3xf32> -> tensor<3xf32>

    tiny.return %1 : tensor<3xf32>
}

tiny.func @cast_noop_erase() -> tensor<3xf32> {
    %0 = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %1 = tiny.cast %0 : tensor<3xf32> -> tensor<3xf32>

    tiny.return %1 : tensor<3xf32>
}

tiny.func @simple_cast() -> tensor<3xf16> {
    %0 = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %1 = tiny.cast %0 : tensor<3xf32> -> tensor<3xf16>

    tiny.return %1 : tensor<3xf16>
}
