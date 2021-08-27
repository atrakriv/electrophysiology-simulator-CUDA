This repo contains four CUDA implementations of a cardiac electrophysiology simulator. This simulator involves solving a PDE and two related ODEs using a 5-point stencil method. 

* V1: A naive GPU parallel simulator that creates a separate kernel for each for-loop in the simulate function. These kernels will all make references to global memory. 
* V2: All kernels are fused into a single kernel. 
* V3: Temporary variables are used to eliminate global memory references. 
* V4: Further Optimization by using shared memory (on-chip memory) on the GPU