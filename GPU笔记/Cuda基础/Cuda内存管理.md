**内存管理：**

当CPU/GPU尝试访问尚未停驻在其上的内存时，会发生页错误并触发其迁移；

**异步预取**：cuda可通过cudaMemPrefetchAsync（dataPointer, size, deviceId）函数将托管内存异步预取到GPU/CPU

**手动内存管理**：cudaMalloc可以直接为GPU分配内存，防止分页错误，但主机无法访问cudaMalloc返回指针；cudaMallocHost可直接为CPU分配内存(新版GPU可直接读取)，固定内存或锁页（过多使用会干扰CPU性能），与GPU之间异步拷贝，使用cudaFreeHost命令释放；cudaMemcy可以在GPU（device）和CPU(host)之间拷贝内存；
```cpp
int *hostAddr, *deviceAddr;
cudaMalloc(&deviceAddr, size); // GPU分配内存
initializeOnHost(hostAddr, N); // cpu上初始化数据
cudaMemcpy(deviceAddr, hostAddr, size, cudaMemcpyHostToDevice); // 初始值从cpu拷贝到gpu
kernelFunc<<<blocks, threads, 0, someStream>>>(deviceAddr, N); //在GPU上多线程计算
cudaMemcpy(hostAddr, deviceAddr, size, cudaMemcpuDeviceToHost); //计算结果从gpu拷贝到cpu
verifyOnHost(hostAddr, N); //在cpu上验证结果
cudaFree(deviceAddr); //释放GPU内存
cudaFreeHost(hostAddr); //释放cpu内存
```


**全局内存**（空间较大，任何线程或块都可使用，存续时间贯穿应用程序的生命周期）

**共享内存**（手动定义的缓存，容量有限，可为同一个线程块内的线程共享，带宽高，核函数执行完毕后释放）

        [https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)

__shared__ float tile[TILE_DIM][TILE_DIM];

        __syncthreads();

![](file:///C:\Users\mi\AppData\Local\Temp\ksohtml14008\wps28.jpg)![](file:///C:\Users\mi\AppData\Local\Temp\ksohtml14008\wps29.jpg)