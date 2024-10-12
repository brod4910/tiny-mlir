// RUN: tiny-opt %s | FileCheck %s


tiny.func @simple_empty() -> tensor<10x10xi8> {
    %0 = tiny.empty [10, 10], i8 : tensor<10x10xi8>

    tiny.return %0 : tensor<10x10xi8>
}

tiny.func @simple_copy() -> tensor<10x10xi8> {
    %0 = tiny.empty[10, 10], i8 : tensor<10x10xi8>
    
    %1 = tiny.copy %0 : (tensor<10x10xi8>) -> tensor<10x10xi8>

    tiny.return %0 : tensor<10x10xi8>
}

tiny.func @simple_view() -> tensor<10x3x784xi8> {
    %0 = tiny.empty [10,3,28,28], i8 : tensor<10x3x28x28xi8>

    %v1 = tiny.view %0[10,3,784] : tensor<10x3x28x28xi8> -> tensor<10x3x784xi8>

    tiny.return %v1 : tensor<10x3x784xi8>
}

tiny.func @simple_contiguous() -> tensor<1x100xi8> {
    %0 = tiny.empty [10, 10], i8 : tensor<10x10xi8>
    %v0 = tiny.view %0[1, 100] : tensor<10x10xi8> -> tensor<1x100xi8>
    %1 = tiny.contiguous %v0 : (tensor<1x100xi8>) -> tensor<1x100xi8>

    tiny.return %1 : tensor<1x100xi8>
}

tiny.func @simple_randn() -> tensor<10x10xf16> {
    %0 = tiny.randn [10,10], f16 : tensor<10x10xf16>

    tiny.return %0 : tensor<10x10xf16>
}

tiny.func @simple_fill() -> tensor<1x100xi8> {
    %0 = tiny.empty [10, 10], i8 : tensor<10x10xi8>
    %v0 = tiny.view %0[1, 100] : tensor<10x10xi8> -> tensor<1x100xi8>
    %1 = tiny.contiguous %v0 : (tensor<1x100xi8>) -> tensor<1x100xi8>
    
    %c = tiny.constant dense<1> : tensor<i8>
    %2 = tiny.fill %1, %c : (tensor<1x100xi8>, tensor<i8>) -> tensor<1x100xi8>

    tiny.return %2 : tensor<1x100xi8>
}