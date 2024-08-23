// RUN: tiny-opt %s -convert-tiny-to-accl -convert-tiny-to-llvm | FileCheck %s

// CHECK-LABEL: @test_log2_llvm
// ./build/tools/tiny-opt tests/Conversion/tiny_to_llvm.mlir  -convert-tiny-func-ops -convert-tiny-to-accl -convert-tiny-elementwise-to-linalg -one-shot-bufferize="bufferize-function-boundaries" -func-bufferize  -convert-linalg-to-affine-loops -lower-affine -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-func-to-llvm -convert-tiny-to-llvm -reconcile-unrealized-casts
module {
    tiny.func @test_log2_llvm(%arg0 : tensor<1x16xf32>) -> (tensor<1x16xf32>) {
        %0 = tiny.log2 %arg0 : tensor<1x16xf32> -> tensor<1x16xf32>
        tiny.return %0 : tensor<1x16xf32>
    }

    tiny.func @test_matmul_llvm(%a: tensor<512x768xf32>, %b: tensor<768x512xf32>) -> tensor<512x512xf32> {
        %av = tiny.view %a[512, 1, 768] : tensor<512x768xf32> -> tensor<512x1x768xf32>
        %bv = tiny.view %b[1, 512, 768] : tensor<768x512xf32> -> tensor<1x512x768xf32>
        %0 = tiny.mul %av, %bv : (tensor<512x1x768xf32>, tensor<1x512x768xf32>) -> tensor<512x512x768xf32>
        %1 = tiny.sum %0 {axis = -1 : si32} : (tensor<512x512x768xf32>) -> tensor<512x512xf32>
        tiny.return %1 : tensor<512x512xf32>
    }
}