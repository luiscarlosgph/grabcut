#ifndef UTIL_H
#define UTIL_H

// Functions from util.cu
cudaError_t TrimapFromRect(Npp8u *alpha, int alpha_pitch, NppiRect rect, int width, int height);

inline int cudaDeviceInit(int argc, const char **argv) {
  int deviceCount;
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));

  if (deviceCount == 0) {
#ifndef NDEBUG
    std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
#endif
    exit(EXIT_FAILURE);
  }

  int dev = findCudaDevice(argc, argv);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
#ifndef NDEBUG
  std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name << std::endl;
#endif

  checkCudaErrors(cudaSetDevice(dev));

  return dev;
}

/*
bool printfNPPinfo(int argc, char *argv[], int cudaVerMajor, int cudaVerMinor) {
  const NppLibraryVersion *libVer = nppGetLibVersion();

  printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
  printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

  bool bVal = checkCudaCapabilities(cudaVerMajor, cudaVerMinor);
  return bVal;
}
*/

#endif
