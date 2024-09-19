// #include <cuda_runtime.h>
// #include "./utils.h"
// #define CUDA_CALL(func)                                       \
//   {                                                           \
//     cudaError_t e = (func);                                   \
//     ICHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) \
//         << "CUDA: " << cudaGetErrorString(e);                 \
//   }

// namespace tvm{
// namespace meta_schedule{

// TVM_REGISTER_GLOBAL("device_api.cuda_reset").set_body_typed([](int device_id) {
//   CUDA_CALL(cudaSetDevice(device_id));
//   CUDA_CALL(cudaDeviceReset());
// });



// }
// }



