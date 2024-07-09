// RUN: tiny-opt %s -convert-tiny-to-accl -convert-tiny-to-llvm | FileCheck %s

// CHECK-LABEL: @test_neg_llvm
module {
    tiny.func @test_neg_llvm(%arg1 : tensor<1x16xf32>) {
        %0 = tiny.log2 %arg1 : tensor<1x16xf32> -> tensor<1x16xf32>
        tiny.return
    }
}