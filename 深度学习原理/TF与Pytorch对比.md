**Pytorch动态图**：根据tensor的**自动微分autograd**的计算过程自动生成（储存了tensor数值和Tensor的微分函数tensor.grad_fn），伴随着新张量或运算的加入不断更新，更加灵活易用。

**Pytorch**: torch.Dataset和DataLoader, torch.nn.module, torch.optimizer定义tensor的forward操作，设置requires_grad = True,然后通过.backward（）来autograde(动态图)，最后optim来更新权重； 
前向计算的赋值运算其实是把当前结果（节点值和微分函数）压栈，反向传播的时候出栈，调用对应的求导操作并把计算好的梯度返回给生成该变量的参数。

**TensorFlow静态图**: 深度学习专用语言（graph和session）,符号式编程+静态图（IR表示能支持多种前端）

a) 通过定义placeholder、Variable（初始化为Tensor,不可变不存在于内存）和OP等构成一张完成计算图Graph（静态图）,先编译图，可以提前做优化（图优化和指定设备上的编译优化）；

![](file:///C:\Users\mi\AppData\Local\Temp\ksohtml14008\wps16.jpg) 

![](file:///C:\Users\mi\AppData\Local\Temp\ksohtml14008\wps17.jpg) 

b) 新建Session实例启动模型运行(target硬件, graph, config)，Session实例会分布式执行Graph，输入数据, 根据优化算法更新Variable，然后返回执行结果即Tensor实例。

executor的cache机制

![](file:///C:\Users\mi\AppData\Local\Temp\ksohtml14008\wps18.jpg)![](file:///C:\Users\mi\AppData\Local\Temp\ksohtml14008\wps19.jpg) 

StreamExecutor用于管理优化流之间的并行（计算流和数据）

![](file:///C:\Users\mi\AppData\Local\Temp\ksohtml14008\wps20.jpg) 

