// RUN: tiny-opt %s | FileCheck %s


tiny.func @simple_load(%arg0: tensor<10x1xi8>) -> tensor<1x1xi8> {
    %slice_012 = tiny.slice[0, 1, 2] : !tiny.slice<0, 1, 2>
    %0 = tiny.load %arg0[%slice_012] : (tensor<10x1xi8>, !tiny.slice<0, 1, 2>) -> tensor<1x1xi8>

    tiny.return %0 : tensor<1x1xi8>
}

tiny.func @multi_slice_load(%arg0: tensor<10x10x10xi8>) -> tensor<5x2x1xi8> {
    %slice_0E2 = tiny.slice[0, -1, 2] : !tiny.slice<0, -1, 2>
    %slice_5E2 = tiny.slice[5, -1, 2] : !tiny.slice<5, -1, 2>
    %slice_9E2 = tiny.slice[9, 10, 1] : !tiny.slice<9, 10, 1>


    %0 = tiny.load %arg0[%slice_0E2, %slice_5E2, %slice_9E2] : (tensor<10x10x10xi8>, !tiny.slice<0, -1, 2>, 
                                                                !tiny.slice<5, -1, 2>, !tiny.slice<9, 10, 1>) -> tensor<5x2x1xi8>

    tiny.return %0 : tensor<5x2x1xi8>
}

tiny.func @simple_store(%arg0: tensor<10x1xi8>, %arg1: tensor<10x1xi8>) -> tensor<10x1xi8> {
    %slice0 = tiny.slice[0] : !tiny.slice<0>
    
    tiny.store %arg0[%slice5], %arg1[%slice0] : (tensor<10x1xi8>, !tiny.slice<5>, tensor<10x1xi8>, !tiny.slice<0>) -> ()

    tiny.return %arg1 : tensor<10x1xi8>
}

tiny.func @multi_slice_store(%arg0: tensor<10x1xi8>, %arg1: tensor<10x1xi8>) -> tensor<10x1xi8> {
    %slice0 = tiny.slice[0] : !tiny.slice<0>

    %slice5 = tiny.slice[5] : !tiny.slice<5>

    tiny.store %arg0[%slice5, %slice0], %arg1[%slice5, %slice0] : (tensor<10x1xi8>, !tiny.slice<5>, !tiny.slice<0>, tensor<10x1xi8>, !tiny.slice<5>, !tiny.slice<0>) -> ()

    tiny.return %arg1 : tensor<10x1xi8>
}