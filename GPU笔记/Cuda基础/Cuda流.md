**Cuda流**

流由按顺序执行的一系列命令（核函数执行和内存传输）构成，cuda默认在默认流中执行核函数；

（1）给定流中的所有操作会顺序执行；（2）不同非默认流间的操作顺序不影响；

（3）默认流具有阻断能力，等待已运行的其他流完成后才运行，自身运行完前会阻碍其他流运行；

![](file:///C:\Users\mi\AppData\Local\Temp\ksohtml14008\wps27.jpg) 

**使用流运行核函数：**cudaStream_t stream; cudaStreamCreat(&stream); // 创建流

        KernelFunc<<<blockNumbers, threadsPerBlock, 0(共享内存动态分配字节数), stream>>> (); // 指定流

        cudaStreamDestroy(stream); // 销毁流

**通过流实现内存分配：**cudaMemcpyAsync可以在CPU和GPU之间异步复制内存（默认情况下对GPU的其他操作阻塞），可通过传递非默认流参数，将内存传输与非默认流中的其他操作并发；

CudaStream_t stream; cudaStreamCreate(&stream); // 创建流

CudaMemcpyAsync(&deviceAddr, &hostAddr, segmentSize, cudaMemcpyHostToDevice, stream); // 内存复制非默认流

cudaStreamDestory(stream); // 销毁流

cuda的内存层次结构：[https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)