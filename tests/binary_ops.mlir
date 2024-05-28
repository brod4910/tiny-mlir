// RUN: tiny-opt %s | FileCheck %s

// CHECK-LABEL: @simple_bool_constant
tiny.func @simple_add() -> tensor<10x10x10xi8> {
    %0 = tiny.constant dense<0> : tensor<10x1x10xi8>
    %1 = tiny.constant dense<1> : tensor<1x10xi8>

    %2 = tiny.add %0, %1 : tensor<?x?x?xi8>

    tiny.return %0 : tensor<i1>
}