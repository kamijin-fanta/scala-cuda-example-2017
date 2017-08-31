import jcuda.driver._
import jcuda.{Pointer, Sizeof}


object MemoryExample {

  def main(args: Array[String]): Unit = {
    JCudaDriver.setExceptionsEnabled(true)
    val ptxFileName = "kernel.ptx"
    JCudaDriver.cuInit(0)
    val device = new CUdevice
    JCudaDriver.cuDeviceGet(device, 0)
    val context = new CUcontext
    JCudaDriver.cuCtxCreate(context, 0, device)

    val module = new CUmodule
    val path = getClass.getResource(ptxFileName).getFile.substring(1) // for win
    JCudaDriver.cuModuleLoad(module, path)
    val function = new CUfunction
    JCudaDriver.cuModuleGetFunction(function, module, "add")

    val numElements = 5000000
    val hostInputA = new Array[Float](numElements)
    val hostInputB = new Array[Float](numElements)
    var i = 0
    val res = i < numElements
    while (i < numElements) {
      hostInputA(i) = i.toFloat
      hostInputB(i) = i.toFloat
      i += 1
    }
    val deviceInputA = new CUdeviceptr
    JCudaDriver.cuMemAlloc(deviceInputA, numElements * Sizeof.FLOAT)
    JCudaDriver.cuMemcpyHtoD(deviceInputA,
                             Pointer.to(hostInputA),
                             numElements * Sizeof.FLOAT)
    val deviceInputB = new CUdeviceptr
    JCudaDriver.cuMemAlloc(deviceInputB, numElements * Sizeof.FLOAT)
    JCudaDriver.cuMemcpyHtoD(deviceInputB,
                             Pointer.to(hostInputB),
                             numElements * Sizeof.FLOAT)
    val deviceOutput = new CUdeviceptr
    JCudaDriver.cuMemAlloc(deviceOutput, numElements * Sizeof.FLOAT)
    val kernelParameters = Pointer.to(Pointer.to(Array[Int](numElements)),
                                      Pointer.to(deviceInputA),
                                      Pointer.to(deviceInputB),
                                      Pointer.to(deviceOutput))
    val blockSizeX = 256
    val gridSizeX = Math.ceil(numElements.toDouble / blockSizeX).toInt
    JCudaDriver.cuLaunchKernel(function,
                               gridSizeX,
                               1,
                               1,
                               blockSizeX,
                               1,
                               1,
                               0,
                               null,
                               kernelParameters,
                               null)
    JCudaDriver.cuCtxSynchronize

    val hostOutput = new Array[Float](numElements)
    JCudaDriver.cuMemcpyDtoH(Pointer.to(hostOutput),
                             deviceOutput,
                             numElements * Sizeof.FLOAT)

    var passed = true
    i = 0
    while (i < numElements && passed) {
      val expected = i + i
      if (Math.abs(hostOutput(i) - expected) > 1e-5) {
        System.out.println(
          "At index " + i + " found " + hostOutput(i) + " but expected " + expected)
        passed = false
      }
      i += 1
    }

    System.out.println(
      "Test " + (if (passed) "PASSED"
                 else "FAILED"))
    // Clean up.
    JCudaDriver.cuMemFree(deviceInputA)
    JCudaDriver.cuMemFree(deviceInputB)
    JCudaDriver.cuMemFree(deviceOutput)
  }

}
