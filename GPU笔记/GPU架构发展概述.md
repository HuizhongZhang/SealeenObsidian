**GPU架构发展**

volta架构一个cuda内核分为fp32运算和int32运算，增加tensor核心张量单元用于深度学习，turing架构配备RTcore光线追踪处理器

empere架构（GA100 SM）第三代tensor核心（计算性能更高，综合数据类型，稀疏加速），更大的L1缓存

(1) **Tensor core**

wmma c++ api

fp16支持8*8*4和16*8*8的矩阵运算，int841，tf32和bf16

稀疏化支持（压缩后的矩阵和坐标）调用kernel（矩阵乘shape和tiling切分shape）