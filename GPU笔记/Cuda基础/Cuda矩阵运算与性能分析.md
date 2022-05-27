**矩阵加法运算：**（非跨步实现）
```cpp
__global__ void gpu_matrix_add (int* a, int* b, int* c_gpu) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < N && y < N) { c[x * N + y] = a[x * N + y] + b[x * N + y]; }
	}

dim3 threads(16,16,1);
dim3 blocks(ceil(N, threads.x), ceil(N, threads.y));
gpu_matirx_add<<<blocks, threads>>>> (a, b, c_gpu);
```

**性能分析：**nsys profile --stats=true ./matrix, 一般cudaMalloc占据大部分的时间，因此显存管理很重要；

![](file:///C:\Users\mi\AppData\Local\Temp\ksohtml14008\wps26.jpg)