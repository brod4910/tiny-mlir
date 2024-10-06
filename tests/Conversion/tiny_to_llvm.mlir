// RUN: tiny-opt %s -convert-tiny-to-accl -convert-tiny-to-llvm | FileCheck %s

// CHECK-LABEL: @test_log2_llvm
// ./build/tools/tiny-opt tests/Conversion/tiny_to_llvm.mlir  -convert-tiny-func-ops -convert-tiny-to-accl -convert-tiny-elementwise-to-linalg -one-shot-bufferize="bufferize-function-boundaries" -func-bufferize  -convert-linalg-to-affine-loops -lower-affine -convert-scf-to-cf -finalize-memref-to-llvm -convert-cf-to-llvm -convert-func-to-llvm -convert-tiny-to-llvm -reconcile-unrealized-casts
module {
    tiny.func @test_log2_llvm(%arg0 : tensor<1x16xf32>) -> (tensor<1x16xf32>) {
        %0 = tiny.log2 %arg0 : tensor<1x16xf32> -> tensor<1x16xf32>
        tiny.return %0 : tensor<1x16xf32>
    }

    tiny.func @test_iadd_llvm(%arg0 : tensor<1x16xi32>, %arg1 : tensor<1x16xi32>) -> (tensor<1x16xi32>) {
        %0 = tiny.add %arg0, %arg1 : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>

        tiny.return %0 : tensor<1x16xi32>
    }

    tiny.func @test_fadd_llvm(%arg0 : tensor<1x16xf32>, %arg1 : tensor<1x16xf32>) -> (tensor<1x16xf32>) {
        %0 = tiny.add %arg0, %arg1 : (tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>

        tiny.return %0 : tensor<1x16xf32>
    }

    tiny.func @test_isub_llvm(%arg0 : tensor<1x16xi32>, %arg1 : tensor<1x16xi32>) -> (tensor<1x16xi32>) {
        %0 = tiny.sub %arg0, %arg1 : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>

        tiny.return %0 : tensor<1x16xi32>
    }

    tiny.func @test_fsub_llvm(%arg0 : tensor<1x16xf32>, %arg1 : tensor<1x16xf32>) -> (tensor<1x16xf32>) {
        %0 = tiny.sub %arg0, %arg1 : (tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>

        tiny.return %0 : tensor<1x16xf32>
    }

    tiny.func @test_imul_llvm(%arg0 : tensor<1x16xi32>, %arg1 : tensor<1x16xi32>) -> (tensor<1x16xi32>) {
        %0 = tiny.mul %arg0, %arg1 : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>

        tiny.return %0 : tensor<1x16xi32>
    }

    tiny.func @test_fmul_llvm(%arg0 : tensor<1x16xf32>, %arg1 : tensor<1x16xf32>) -> (tensor<1x16xf32>) {
        %0 = tiny.mul %arg0, %arg1 : (tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>

        tiny.return %0 : tensor<1x16xf32>
    }

    tiny.func @test_fdiv_llvm(%arg0 : tensor<1x16xf32>, %arg1 : tensor<1x16xf32>) -> (tensor<1x16xf32>) {
        %0 = tiny.div %arg0, %arg1 : (tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>

        tiny.return %0 : tensor<1x16xf32>
    }

    tiny.func @test_idiv_llvm(%arg0 : tensor<1x16xi32>, %arg1 : tensor<1x16xi32>) -> (tensor<1x16xi32>) {
        %0 = tiny.div %arg0, %arg1 : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>

        tiny.return %0 : tensor<1x16xi32>
    }

    tiny.func @test_cmplt_llvm(%arg0 : tensor<1x16xf32>, %arg1 : tensor<1x16xf32>) -> (tensor<1x16xi1>) {
        %0 = tiny.cmplt %arg0, %arg1 : (tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xi1>

        tiny.return %0 : tensor<1x16xi1>
    }

    tiny.func @test_cmpne_llvm(%arg0 : tensor<1x16xf32>, %arg1 : tensor<1x16xf32>) -> (tensor<1x16xi1>) {
        %0 = tiny.cmpne %arg0, %arg1 : (tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xi1>

        tiny.return %0 : tensor<1x16xi1>
    }
    
    tiny.func @test_imaximum_llvm(%arg0 : tensor<1x16xi32>, %arg1 : tensor<1x16xi32>) -> (tensor<1x16xi32>) {
        %0 = tiny.maximum %arg0, %arg1 : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>

        tiny.return %0 : tensor<1x16xi32>
    }

    tiny.func @test_fmaximum_llvm(%arg0 : tensor<1x16xf32>, %arg1 : tensor<1x16xf32>) -> (tensor<1x16xf32>) {
        %0 = tiny.maximum %arg0, %arg1 : (tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>

        tiny.return %0 : tensor<1x16xf32>
    }

    tiny.func @test_mod_llvm(%arg0 : tensor<1x16xi32>, %arg1 : tensor<1x16xi32>) -> (tensor<1x16xi32>) {
        %0 = tiny.mod %arg0, %arg1 : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>

        tiny.return %0 : tensor<1x16xi32>
    }

    tiny.func @test_xor_llvm(%arg0 : tensor<1x16xi32>, %arg1 : tensor<1x16xi32>) -> (tensor<1x16xi32>) {
        %0 = tiny.xor %arg0, %arg1 : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>

        tiny.return %0 : tensor<1x16xi32>
    }

    tiny.func @test_shl_llvm(%arg0 : tensor<1x16xi32>, %arg1 : tensor<1x16xi32>) -> (tensor<1x16xi32>) {
        %0 = tiny.shl %arg0, %arg1 : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>

        tiny.return %0 : tensor<1x16xi32>
    }

    tiny.func @test_shr_llvm(%arg0 : tensor<1x16xi32>, %arg1 : tensor<1x16xi32>) -> (tensor<1x16xi32>) {
        %0 = tiny.shr %arg0, %arg1 : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>

        tiny.return %0 : tensor<1x16xi32>
    }

    // tiny.func @test_matmul_llvm(%a: tensor<512x768xf32>, %b: tensor<768x512xf32>) -> tensor<512x512xf32> {
    //     %av = tiny.view %a[512, 1, 768] : tensor<512x768xf32> -> tensor<512x1x768xf32>
    //     %bv = tiny.view %b[1, 512, 768] : tensor<768x512xf32> -> tensor<1x512x768xf32>
    //     %0 = tiny.mul %av, %bv : (tensor<512x1x768xf32>, tensor<1x512x768xf32>) -> tensor<512x512x768xf32>
    //     %1 = tiny.sum %0 {axis = -1 : si32} : (tensor<512x512x768xf32>) -> tensor<512x512xf32>
    //     tiny.return %1 : tensor<512x512xf32>
    // }
}