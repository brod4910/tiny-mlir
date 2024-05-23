// RUN: tiny-opt %s -split-input-file -convert-tiny-to-accl='target=cuda:80 num-warps=2' | FileCheck %s

#layout0 = #accl.cta_layout<{threadBlockTile = [1, 1, 1], warpTile = [2, 1, 1], threadTile = [8, 1, 1]}>
module attributes { "accl.num-warps" = 16 : i32, "accl.threads-per-warp" = 16: i32} {
    tiny.func @test_layouts() {
        // CHECK: #[[layout0:.*]] = #accl.cta_layout<{threadBlockTile = [1, 1, 1], warpTile = [2, 1, 1], threadTile = [8, 1, 1]}>
        // CHECK: module attributes {"tiny.num-warps" = 16 : i32, "tiny.threads-per-warp" = 16 : i32} {{.*}}
        %c0 = tiny.constant dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]> : tensor<1x16xi8, #layout0>
        // CHECK: (tensor<1x16xi8, #[[layout0]]>) -> (tensor<1x16xi8, #[[layout0]]>)
        %0 = tiny.neg %c0 : tensor<1x16xi8> -> tensor<1x16xi8>
        tiny.return
    }
}