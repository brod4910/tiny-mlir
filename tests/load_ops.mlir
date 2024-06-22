// RUN: tiny-opt %s | FileCheck %s


tiny.func @simple_empty() -> tensor<10x10xi8> {
    %s = tiny.shape[10,10], i8 : !tiny.shape<10x10xi8>
    %0 = tiny.empty %s : (!tiny.shape<10x10xi8>) -> tensor<10x10xi8>

    tiny.return %0 : tensor<10x10xi8>
}

tiny.func @simple_copy() -> tensor<10x10xi8> {
    %s = tiny.shape[10,10], i8 : !tiny.shape<10x10xi8>
    %0 = tiny.empty %s : (!tiny.shape<10x10xi8>) -> tensor<10x10xi8>
    
    %1 = tiny.copy %0 : (tensor<10x10xi8>) -> tensor<10x10xi8>

    tiny.return %0 : tensor<10x10xi8>
}

tiny.func @simple_view() -> tensor<3x28x28xi8> {
    %s = tiny.shape[10,3,28,28], i8 : !tiny.shape<10x3x28x28xi8>
    %0 = tiny.empty %s : (!tiny.shape<10x3x28x28xi8>) -> tensor<10x3x28x28xi8>
    
    %slice_0 = tiny.slice[0] : !tiny.slice<0>
    %slice_1 = tiny.slice[0, 3] : !tiny.slice<0, 3>
    %slice_2 = tiny.slice[0, 28] : !tiny.slice<0, 28>

    %v1 = tiny.view %0[%slice_0, %slice_1, %slice_2, %slice_2] : (tensor<10x3x28x28xi8>, 
                                                                  !tiny.slice<0>, 
                                                                  !tiny.slice<0, 3>, 
                                                                  !tiny.slice<0, 28>, 
                                                                  !tiny.slice<0, 28>) -> tensor<3x28x28xi8>

    tiny.return %v1 : tensor<3x28x28xi8>
}

tiny.func @simple_contiguous() -> tensor<10x10xi8> {
    %s = tiny.shape[10,10], i8 : !tiny.shape<10x10xi8>
    %0 = tiny.empty %s : (!tiny.shape<10x10xi8>) -> tensor<10x10xi8>
    
    %1 = tiny.copy %0 : (tensor<10x10xi8>) -> tensor<10x10xi8>

    tiny.return %0 : tensor<10x10xi8>
}