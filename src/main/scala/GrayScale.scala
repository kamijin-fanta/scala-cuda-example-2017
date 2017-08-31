import java.awt.image.{BufferedImage, DataBufferByte, Raster}
import java.io.File
import javax.imageio.ImageIO

import jcuda.driver._
import jcuda.{Pointer, Sizeof}

object GrayScale {
  def main(args: Array[String]): Unit = {
    val image: BufferedImage = ImageIO.read(new File(getClass.getResource("ex.tif").toURI))
    val raster = image.getRaster.getDataBuffer.asInstanceOf[DataBufferByte]
    println(s"Byte: ${raster.getData().length}  Resolution: ${image.getWidth()}x${image.getHeight()} = ${image.getHeight()*image.getWidth()}")

    val deviceInput = new CUdeviceptr
    val deviceOutput = new CUdeviceptr
    try {
      println(System.currentTimeMillis())

      JCudaDriver.setExceptionsEnabled(true)
      JCudaDriver.cuInit(0)
      val device = new CUdevice
      JCudaDriver.cuDeviceGet(device, 0)
      val context = new CUcontext
      JCudaDriver.cuCtxCreate(context, 0, device)

      val hostInput = raster.getData
      val width = image.getWidth
      val height = image.getHeight
      val totalPixcel = width * height

      val hostInputSize = hostInput.length * Sizeof.BYTE
      JCudaDriver.cuMemAlloc(deviceInput, hostInput.length * Sizeof.BYTE)
      JCudaDriver.cuMemAlloc(deviceOutput, hostInput.length * Sizeof.BYTE)
      JCudaDriver.cuMemAlloc(deviceInput, hostInputSize)
      JCudaDriver.cuMemcpyHtoD(deviceInput, Pointer.to(hostInput), hostInputSize)
      JCudaDriver.cuMemcpyHtoD(deviceInput, Pointer.to(hostInput), hostInput.length)

      println(System.currentTimeMillis())


      val ptxFileName = "kernel.ptx"
      val module = new CUmodule
      val path = getClass.getResource(ptxFileName).getFile.substring(1) // for win
      println(s"ptx: ${path}")
      JCudaDriver.cuModuleLoad(module, path)
      val function = new CUfunction
      JCudaDriver.cuModuleGetFunction(function, module, "RGBToYKernel")

      val kernelParameters: Pointer = Pointer.to(
        Pointer.to(deviceInput),
        Pointer.to(deviceOutput),
        Pointer.to(Array[Int](width)),
        Pointer.to(Array[Int](height))
      )

      val blockSize = 32
      val gridSize = Math.ceil(totalPixcel / blockSize).toInt
      JCudaDriver.cuLaunchKernel(function, gridSize, 1, 1, blockSize, 1, 1, 0, null, kernelParameters, null)
      JCudaDriver.cuCtxSynchronize

      val hostOutput = new Array[Byte](hostInputSize)
      JCudaDriver.cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput, hostInput.length)
      println(System.currentTimeMillis())


      val newBuffer = new DataBufferByte(hostOutput, hostInputSize)
      val newRaster = Raster.createWritableRaster(image.getRaster.getSampleModel, newBuffer, null)
      val newImage = new BufferedImage(image.getWidth, image.getHeight, BufferedImage.TYPE_INT_BGR)
      newImage.setData(newRaster)
      ImageIO.write(newImage, "tif", new File("./hoge.tif"))

    } catch {
      case ex: Throwable => ex.printStackTrace()
    } finally {
      JCudaDriver.cuMemFree(deviceInput)
      JCudaDriver.cuMemFree(deviceOutput)
    }
  }
}
