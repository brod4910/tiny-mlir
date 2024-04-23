// RUN: tiny-opt %s | FileCheck %s
tiny.func @simple_cast() -> tensor<3xf16> {
    %0 = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %1 = tiny.cast %0 : tensor<3xf32> -> tensor<3xf16>

    tiny.return %1 : tensor<3xf16>
}

tiny.func @simple_noop() -> () {
    %0 = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    %1 = tiny.noop %0 : tensor<3xf32> -> tensor<3xf32>

    tiny.return
}