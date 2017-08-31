nvcc -ptx -arch compute_30 -code compute_30,sm_30 -o JCudaVectorAddKernel.ptx JCudaVectorAddKernel.cu
