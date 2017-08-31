import java.awt.image.{BufferedImage, DataBufferByte, Raster}
import java.io.File
import javax.imageio.ImageIO

import jcuda.driver._
import jcuda.{Pointer, Sizeof}

object Image {
  def main(args: Array[String]): Unit = {
    val image: BufferedImage = ImageIO.read(new File(getClass.getResource("ex.tif").toURI))
    val raster = image.getRaster.getDataBuffer.asInstanceOf[DataBufferByte]
    val elems = (0 to 10).map(raster.getElem _)
    println(raster.getData().length, s"${image.getWidth()}x${image.getHeight()} = ${image.getHeight()*image.getWidth()}")

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
      for (x <- 0 to (image.getHeight() - 1)) {
        hostInput(x * width * 3 + width / 2 * 3) = 0xff.asInstanceOf[Byte]
      }

      val hostInputSize = hostInput.length * Sizeof.BYTE
      val pitch = image.getWidth * 3L
      // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a
      JCudaDriver.cuMemAllocPitch(deviceInput, Array(pitch), image.getWidth * 3 * Sizeof.BYTE, height, 16)
      JCudaDriver.cuMemAllocPitch(deviceOutput, Array(pitch), image.getWidth * 3 * Sizeof.BYTE, height, 16)
      JCudaDriver.cuMemAlloc(deviceInput, hostInputSize)
      JCudaDriver.cuMemcpyHtoD(deviceInput, Pointer.to(hostInput), hostInputSize)
      val toDeviceOption = new CUDA_MEMCPY2D()
      toDeviceOption.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST
      toDeviceOption.srcHost = Pointer.to(hostInput)
      toDeviceOption.srcPitch = pitch
      toDeviceOption.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_DEVICE
      toDeviceOption.dstDevice = deviceInput
      toDeviceOption.dstPitch = pitch
      toDeviceOption.Height = height
      toDeviceOption.WidthInBytes = image.getWidth * 3
      JCudaDriver.cuMemcpy2D(toDeviceOption)
      //      JCudaDriver.cuMemcpyHtoD(deviceInput, Pointer.to(hostInput), hostInput.length)

      println(System.currentTimeMillis())


      val ptxFileName = "kernel.ptx"
      val module = new CUmodule
      val path = getClass.getResource(ptxFileName).getFile.substring(1) // for win
      println(s"ptx: ${path}")
      JCudaDriver.cuModuleLoad(module, path)
      val function = new CUfunction
      JCudaDriver.cuModuleGetFunction(function, module, "BlurKernel")


      val genOpacityMap = (round: Int) => for {
        y <- (-round to round)
        x <- (-round to round)
      } yield (x, y, Math.max((round - Math.sqrt(Math.pow(x, 2) + Math.pow(y, 2))) / round, 0))
      val opacityTable = genOpacityMap(3).flatMap {
        case (x, y, o) => List(x.toByte, y.toByte, (o * 255).toInt.toByte)
      }.toArray
      //      println(opacityTable.sliding(3,3).map(_.mkString(",")).mkString("\n"))

      val tablePointer = new CUdeviceptr
      val tableLenPointer = new CUdeviceptr
      JCudaDriver.cuModuleGetGlobal(tablePointer, Array(Sizeof.BYTE * opacityTable.length), module, "TABLE")
      JCudaDriver.cuModuleGetGlobal(tableLenPointer, Array(Sizeof.INT), module, "TABLE_LEN")
      JCudaDriver.cuMemcpyHtoD(tablePointer, Pointer.to(opacityTable), Sizeof.BYTE * opacityTable.length)
      JCudaDriver.cuMemcpyHtoD(tableLenPointer, Pointer.to(Array(opacityTable.length)), Sizeof.INT)
      var hoge: Array[Int] = new Array(1)
      JCudaDriver.cuMemcpyDtoH(Pointer.to(hoge), tablePointer, Sizeof.INT * 1)

      val kernelParameters: Pointer = Pointer.to(
        Pointer.to(deviceInput),
        Pointer.to(deviceOutput),
        Pointer.to(Array[Int](width)),
        Pointer.to(Array[Int](height)),
        Pointer.to(Array[Int](pitch.toInt))
      )

      val blockSize = 16
      val gridSizeX = Math.ceil(pitch.toDouble / blockSize).toInt
      val gridSizeY = Math.ceil(image.getHeight.toDouble / blockSize).toInt
      JCudaDriver.cuLaunchKernel(function,
        gridSizeX,
        gridSizeY,
        //        1,
        //        1,
        1,
        blockSize,
        //        16,
        blockSize,
        //        1,
        1,
        0,
        null,
        kernelParameters,
        null)
      JCudaDriver.cuCtxSynchronize

      //      JCudaDriver.cuMemcpyHtoD(deviceInput, Pointer.to(hostInput), hostInputSize)
      val hostOutput = new Array[Byte](hostInputSize)
      val toHostOption = new CUDA_MEMCPY2D()
      toHostOption.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_DEVICE
      toHostOption.srcDevice = deviceOutput
      toHostOption.srcPitch = pitch
      toHostOption.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST
      toHostOption.dstHost = Pointer.to(hostOutput)
      toHostOption.dstPitch = pitch
      toHostOption.Height = image.getHeight
      toHostOption.WidthInBytes = image.getWidth * 3
      JCudaDriver.cuMemcpy2D(toHostOption)
      JCudaDriver.cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput, hostInput.length)
      println(System.currentTimeMillis())


      //      JCudaDriver.cuMemcpyDtoH(Pointer.to(hostOutput), deviceInput, hostInputSize)

      val newBuffer = new DataBufferByte(hostOutput, hostInputSize)
      val newRaster = Raster.createWritableRaster(image.getRaster.getSampleModel, newBuffer, null)
      val newImage = new BufferedImage(image.getWidth, image.getHeight, BufferedImage.TYPE_INT_BGR)
      newImage.setData(newRaster)
      ImageIO.write(newImage, "tif", new File("./hoge.tif"))

    } catch {
      case ex => ex.printStackTrace()
    } finally {
      JCudaDriver.cuMemFree(deviceInput)
      JCudaDriver.cuMemFree(deviceOutput)
    }
  }
}
