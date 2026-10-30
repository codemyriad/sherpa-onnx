{
  global:
    SherpaOnnx*;
    # For offline punctuation.
    SherpaOffline*;
    # Cassini: Linux x86_64 CUDA builds only; see cassini-cuda-props-cache.cc.
    cudaGetDeviceProperties;
    cudaGetDeviceProperties_v2;
  local:
    *;
};
