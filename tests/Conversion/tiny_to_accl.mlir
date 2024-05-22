// RUN: tiny-opt %s -split-input-file -convert-tiny-to-accl='target=cuda:80 num-warps=2' | FileCheck %s

module attributes { "tiny.num-warps" = 16 : i8, "tiny.threads-per-warp" = 16: i8} {
    tiny.func @test_layouts() {
        %c0 = tiny.constant dense<[4, 16]> : tensor<4x16xi8>
        
        tiny.return
    }
}