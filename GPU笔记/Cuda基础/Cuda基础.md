l **Cuda基础

**Gpu执行步骤：**执行完要和cpu同步，cudaDeviceSynchronized();

![](file:///C:\Users\mi\AppData\Local\Temp\ksohtml14008\wps21.jpg) 

**编译指令**：nvcc -arch=sm_75 -o hello-gpu hello-gpu.cu -run

**gpu函数定义：**__global__ void gpu_func() {}

**gpu核函数调用**；gpu_func<<<16（线程块的数量）,256（每个线程块包含的线程数量）>>>();

**线程、块和网络：**gridDim.x表示网格的块数，blockIdx.x表示当前块的索引，blockDim.x表示块的线程数，threadIdx.x表示当前块中线程的索引；当前块的当前线程总索引为blockIdx.x * blockDim.x + threadIdx.x;

**显存分配：**malloc只能被cpu使用，cudaMallocManaged可用于cpu和gpu, int *a; cudaMallocManaged(&a, size); cudaFree(a);

![](file:///C:\Users\mi\AppData\Local\Temp\ksohtml14008\wps22.jpg)![](file:///C:\Users\mi\AppData\Local\Temp\ksohtml14008\wps23.jpg) 

**跨步循环：**threadI = blockIdx.x * blockDim.x + threadIdx.x;  stride = gridDim.x * blockDim.x;

for(int i = threadI; i < N; i += stride)  { a[i] *= 2; }

**异常处理：**cudaError_t err = cudaMallocManaged(&a, size);核函数可使用err = cudaGetLastError();(是否为cudaSuccess)

![](file:///C:\Users\mi\AppData\Local\Temp\ksohtml14008\wps24.jpg)![](file:///C:\Users\mi\AppData\Local\Temp\ksohtml14008\wps25.jpg) 

