# Tiny MLIR

## Motivation
In an effort to learn ML Compilers, MLIR, and get closer to ML algorithmic implementation, I've created Tiny MLIR. An ML compiler with inspiration from TinyGrad.

Similar to how TinyGrad was created, my main goal is to learn. Plain and simple.

The repo will attempt to model TinyGrad as close as possible but judging from current work done, there are already differences and peculiarities.

## Goals
Other than learning, the goal is to create an MLIR backend using the low-level ops from Tinygrad and create an autograd Tensor Library using a Python frontend.

## Roadmap
- MLIR backend - incomplete
    - ODS Op, Type, and Attribute Specification
        - Constanly being updated as backend gets more developed
        - General scaffolding in place
    - Conversion to Generic Accelerator
        - Conversion works but haven't developed this part yet since lowering to LLVM seems more important
    - Lowering to LLVM
        - ~~Unary (Elementwise)~~
        - Binary (Elementwise Broadcastable)
        - Ternary (Elementwise Broadcastable)
        - Reduction (Reducer)
        - Load (Creation)
        - Buffer (Load & Stores)
    - Optimizations
        - I'm not touching this part yet since lowering to LLVM is more important and knowing myself, I'll never come back from this world once I've stepped in.

- Autograd Tensor Library - Not Started