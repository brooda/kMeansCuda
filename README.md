# Implementation of parallel K-means algorithm for CUDA

If was an university project.

## Content of directory
* classic.h - classical, sequential approach,
* parallel.h - parallel approach for CUDA,
* main.cu - entry point for program. Runs classic and parallel versions and compares running times

## Input and output data (for experiments)
Input data can be generated by *CreateInput.ipynb*. Then to check that data were indeed well classified, it is possible to run *Visualize results.ipynb*.

## Program is compiled to executable file: cuda