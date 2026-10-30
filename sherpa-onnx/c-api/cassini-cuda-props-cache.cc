// sherpa-onnx/c-api/cassini-cuda-props-cache.cc
//
// Cassini fork only (see CASSINI.md): a per-device cache for
// cudaGetDeviceProperties(_v2), compiled into libsherpa-onnx-c-api.so for
// Linux x86_64 CUDA builds only (see CMakeLists.txt).
//
// Why. ONNX Runtime 1.28.2, which this tree pins, bundles cuDNN-frontend
// v1.24.0 (ORT 1.27.1 had v1.12.0). Its Graph::check_support() calls
// Context::populate_sm_version_from_device(), which calls
// cudaGetDeviceProperties(). ORT's CUDA Conv
// (onnxruntime/core/providers/cuda/nn/conv.cc, CreateCudnnFeExecutionPlan)
// builds a new frontend graph per Conv node whenever that node's input shape
// changes. The Parakeet encoder has 77 Conv nodes and every VAD segment has a
// new length, so every segment pays 77 calls of about 1 ms each (almost all
// kernel time on the RTX 5060 we measured): GPU transcription got 22-42%
// slower. Device properties cannot change while a process runs, so the first
// successful answer per device is kept and copied out afterwards.
//
// How it binds. A program links this library (DT_NEEDED), so its definitions
// are in the global symbol scope ahead of the libcudart.so.12 that ORT's CUDA
// provider brings in later, and the provider's cudaGetDeviceProperties_v2
// reference resolves here. If this library is itself dlopen()ed RTLD_LOCAL,
// nothing binds here and the cache simply never runs. The real functions are
// looked up through the libcudart.so.12 handle, whose lookup scope is
// libcudart and its own dependencies, never RTLD_DEFAULT or RTLD_NEXT, which
// could find these very definitions again.
//
// ABI. The build has no CUDA headers, so the struct is opaque here and its
// size is a constant of the CUDA 12 ABI: _v2 takes the 1032-byte CUDA 12
// cudaDeviceProp (sizeof with every 12.0 to 12.9 header); the unversioned
// entry point keeps the CUDA 11 layout and writes 728 bytes (measured on
// libcudart 12.9). The real call fills a zeroed buffer of ours, and exactly
// that many bytes are copied out, never more than a caller of that ABI
// provides. The cache engages only when cudaRuntimeGetVersion() reports 12.x;
// otherwise every call passes straight through. Only successful results are
// cached. CASSINI_CUDA_PROPS_CACHE=0 forces pass-through, for debugging.
//
// Remove when the pinned ORT bundles cuDNN-frontend >= v1.32.0 (it reads the
// SM version with cudaDeviceGetAttribute) or ORT calls Graph::set_sm_version()
// on its Conv graphs itself.

#include <dlfcn.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace {

using GetPropsFn = int (*)(void *prop, int device);
using RuntimeVersionFn = int (*)(int *version);

constexpr int kCudaSuccess = 0;
// cudaErrorSharedObjectSymbolNotFound. Returned only if libcudart.so.12 or
// the symbol cannot be found, which cannot happen for a caller that was
// itself linked against libcudart.so.12.
constexpr int kCudaErrorSymbolNotFound = 302;
constexpr int kMaxDevices = 64;
constexpr std::size_t kPropSizeV2 = 1032;     // CUDA 12 cudaDeviceProp
constexpr std::size_t kPropSizeLegacy = 728;  // CUDA 11 layout

struct Slot {
  std::atomic<bool> ready{false};
  std::mutex mutex;
  alignas(8) unsigned char props[kPropSizeV2];
};

struct Entry {
  const char *name;
  std::size_t size;
  GetPropsFn real;
  Slot slots[kMaxDevices];
};

Entry g_v2{"cudaGetDeviceProperties_v2", kPropSizeV2, nullptr, {}};
Entry g_legacy{"cudaGetDeviceProperties", kPropSizeLegacy, nullptr, {}};
bool g_cache = false;  // the cache may engage; set once, under g_init
std::once_flag g_init;

void Init() {
  void *cudart = dlopen("libcudart.so.12", RTLD_NOW | RTLD_NOLOAD);
  if (!cudart) cudart = dlopen("libcudart.so.12", RTLD_NOW);
  if (!cudart) return;
  g_v2.real = reinterpret_cast<GetPropsFn>(dlsym(cudart, g_v2.name));
  g_legacy.real = reinterpret_cast<GetPropsFn>(dlsym(cudart, g_legacy.name));

  const char *env = std::getenv("CASSINI_CUDA_PROPS_CACHE");
  if (env && std::strcmp(env, "0") == 0) return;

  auto runtime_version = reinterpret_cast<RuntimeVersionFn>(
      dlsym(cudart, "cudaRuntimeGetVersion"));
  int version = 0;
  if (!runtime_version || runtime_version(&version) != kCudaSuccess) return;
  g_cache = version >= 12000 && version < 13000;
}

int GetProps(Entry *e, void *prop, int device) {
  std::call_once(g_init, Init);
  if (!e->real) return kCudaErrorSymbolNotFound;
  if (!g_cache || !prop || device < 0 || device >= kMaxDevices) {
    return e->real(prop, device);
  }

  Slot &slot = e->slots[device];
  if (!slot.ready.load(std::memory_order_acquire)) {
    std::lock_guard<std::mutex> lock(slot.mutex);
    if (!slot.ready.load(std::memory_order_relaxed)) {
      // Fill our own zeroed buffer, so the bytes the runtime leaves unwritten
      // (the reserved tail of the _v2 struct) are the same for every caller.
      // A failure (bad device, no driver) is returned and not cached.
      std::memset(slot.props, 0, sizeof(slot.props));
      int ret = e->real(slot.props, device);
      if (ret != kCudaSuccess) return ret;
      slot.ready.store(true, std::memory_order_release);
    }
  }
  std::memcpy(prop, slot.props, e->size);
  return kCudaSuccess;
}

}  // namespace

// Exported through sherpa-onnx-symbols-c-cuda.lds. The CUDA signatures are
// cudaError_t (an int-sized enum) and struct cudaDeviceProp *.
extern "C" __attribute__((visibility("default"))) int
cudaGetDeviceProperties_v2(void *prop, int device) {
  return GetProps(&g_v2, prop, device);
}

extern "C" __attribute__((visibility("default"))) int cudaGetDeviceProperties(
    void *prop, int device) {
  return GetProps(&g_legacy, prop, device);
}
