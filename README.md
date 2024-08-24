# Tiny MLIR

## Motivation
In an effort to learn ML Compilers, MLIR, and get closer to ML algorithmic implementation, I've created Tiny MLIR. An ML compiler with inspiration from TinyGrad.

Similar to how TinyGrad was created, my main goal is to learn. Plain and simple.

## Goals
Other than learning, the goal is to create an MLIR backend using the low-level ops from Tinygrad and create an autograd Tensor Library using a Python frontend.

## Roadmap
- MLIR backend - incomplete
    - ODS Op, Type, and Attribute Specification
        - Constanly being updated as backend gets more developed
    - Conversion to Generic Accelerator
        - Conversion works but haven't developed this part yet since lowering to LLVM seems more important
    - Lowering to LLVM
        - ~~Unary (Elementwise)~~
        - Binary (Elementwise Broadcastable)
        - Ternary (Elementwise Broadcastable)
        - Reduction (Reducer)
        - Load (Creation)
        - Buffer (Load & Stores)

- Autograd Tensor Library - Not Started