**Cuda/cuDNN/tensorRT;**

cuda优化方法（线程并行，共享内存，tensor core）

**cuDNN 8的主要特征****:**

a）适用于所有常见卷积的Tensor Core加速，包括2D，3D，分组，深度可分离以及使用NHWC和NCHW输入和输出进行扩张；

b）针对计算机视觉和语音模型的优化内核，包括ResNet，ResNext，SSD，MaskRCNN，Unet，VNet，BERT，GPT-2，Tacotron2和WaveGlow；

c）支持FP32，FP16和TF32浮点格式以及INT8和UINT8整数格式；

d）4d张量的任意维排序，跨距和子区域意味着可以轻松集成到任何神经网络实现中加速任何CNN架构上的融合操作

**TensorRT**

cuda神经网络推理库，核心是c++库，构建阶段生成推理引擎

图优化，消除dropout等，垂直合并conv bn relu，水平合并相同操作

量化和精度校准

内核自动调整

内存优化（动态张量显存）

多流执行并行化