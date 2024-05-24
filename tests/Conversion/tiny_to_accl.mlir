// RUN: tiny-opt %s -convert-tiny-to-accl='target=cuda:80 num-warps=2' | FileCheck %s

// CHECK-LABEL: @test_simple_layout
// CHECK: tiny.constant {{.*}} : [[TENSOR0:.*]]
// CHECK: tiny.neg {{.*}} : [[TENSOR0]] -> [[TENSOR0]]
module attributes {"accl.num-warps" = 16 : i32, "accl.threads-per-warp" = 16 : i32} {
    tiny.func @test_simple_layout() {
        %c0 = tiny.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]> : tensor<1x16xi8>
        %0 = tiny.neg %c0 : tensor<1x16xi8> -> tensor<1x16xi8>
        tiny.return
    }
}

// CHECK-LABEL: @test_expanded_dense
// CHECK: tiny.constant {{.*}} : [[TENSOR1:.*]]
// CHECK: tiny.exp2 {{.*}} : [[TENSOR1]] -> [[TENSOR1]]
module attributes {"accl.num-warps" = 16 : i32, "accl.threads-per-warp" = 16 : i32} {
    tiny.func @test_expanded_dense() {
        %c0 = tiny.constant dense<0> : tensor<3x224x224xi8>
        %0 = tiny.exp2 %c0 : tensor<3x224x224xi8> -> tensor<3x224x224xi8>
        tiny.return
    }
}
