# User Cookbook

-------------------------

## 如何控制NAS算法过程？

可以透过修改配置参数，让NAS工程可以运行在不同环境执行不同任务。

下列为用户相关的参数：

+ nas_main 总控参数
  + num_gpu 运行环境GPU个数
  + block_num 堆叠网络块数量
  + add_data_per_round 每一轮竞赛增加数据大小
  + add_data_for_winner 竞赛胜利者的训练数据集大小(-1代表使用全部数据)
  + repeat_search 模块重复次数
+ enum 穷举模块参数
  + depth 枚举的网络结构的深度
  + width 枚举的网络结构的支链个数
  + max_depth 约束支链上节点的最大个数
+ spl 采样参数
  + skip_max_dist 最大跨层长度
  + skip_max_num 最大跨层个数

> 所有工程相关配置参数都定义于 [nas_config.py](../nas_config.py)
>
> 如果想知道更多参数细节，请参考 [interface.md](interface.md)

## 如何找到工程日志文件？

所有日志文件都放在工程文件夹memory：

1. 评估过程 evaluator_log.txt
2. 子进程 subproc_log.txt
3. 网络信息 network_info.txt
4. 总控 nas_log.txt

> 日志纪录对象 **Logger** 提供全工程统一的日志纪录接口
> 如果需要纪录更多更复杂的日志，可以重新实现 **Logger**。
> 具体使用说明，请参考 [interface.md](interface.md)

## 如何配置新的节点操作？

如果想要自定义特定的网络节点操作配置，请完成以下几点：

1. 修改 `NAS_CONFIG['spl']['space']` 搜索空间
2. 修改 `Evaluator._make_layer` 转换成具体网络节点，并且实现自定义操作的函数

### 具体范例

打算加入新的操作**Separable Convolution**到神经网络，其中包含几个参数**filter_size**, **kernel_size**, **activation**

```json
"sep_conv": {
  "filter_size": [
    [32, 48, 64],
    [48, 64, 128],
    [64, 128, 192],
    [128, 192, 256]
  ],
  "kernel_size": [
    1,
    3,
    5
  ],
  "activation": [
    "relu",
    "leakyrelu",
    "relu6"
  ]
}
```

> **注意**：当参数是**1维列表**，代表参数的搜索空间；当参数是**2维列表**，代表**模块**的参数搜索空间(按照列表次序)

接下来，只需要简单地放入配置参数文件 [nas_config.json](../nas_config.json)/spl/space底下

```python
"spl":{
    "pool_switch": 0,
    "skip_max_dist": 4,
    "skip_max_num": 3,
    "space": {,
      "sep_conv": {
        "filter_size": [
          [32, 48, 64],
          [48, 64, 128],
          [64, 128, 192],
          [128, 192, 256]
        ],
        "kernel_size": [
          1,
          3,
          5
        ],
        "activation": [
          "relu",
          "leakyrelu",
          "relu6"
        ]
      }
    }
  }
```

最后，在`Evaluator._make_layer`，给出配置参数的具体操作方法

```python
def _make_layer(self, inputs, cell, node, train_flag):
    if cell.type == 'sep_conv':
        layer = self._makeconv(inputs, cell, node, train_flag)
    else:
        assert False, "Wrong cell type!"
    return layer
def _makesep_conv(self, inputs, hplist, node, train_flag):
    # TODO
    return sep_conv_layer
```

> **Separable Convolution** 已经在NAS工程中实现了，可以用来参考。

## 如何修改数据集？

1. 将数据集放在data文件夹底下
   > 也可以更改 `NAS_CONFIG['eva']['dataset_path']` 的数据集路径，或是直接修改 `Dataset.data_path` (定义在evalutor.py或evalutor_user.py)
2. 根据具体任务，设置数据集参数：
    1. IMAGE_SIZE 图片尺寸
    2. NUM_CLASSES 分类数量/输出张量的最后一维大小
    3. NUM_EXAMPLES_FOR_TRAIN 训练数据集大小
    4. NUM_EXAMPLES_FOR_EVAL 评估数据集大小
3. 实现 `Dataset.input` 方法，返回数据集：
    1. train_data 训练样本
    2. train_label 训练标签
    3. valid_data 评估样本
    4. valid_label 评估标签
    5. test_data 测试样本
    6. test_label 测试标签
4. (可选) 实现数据增强方法 `Dataset.process`

> 数据集类型 `Dataset` 具体细节可以参考下方说明，以及 [evaluator.py](../evaluator.py) 中读入cifar-10数据集的范例。


## 如何实现针对具体任务的评估方法？

请实现`Evalutor._eval`方法。我们已经给出方法的模版，以及评估需要的参数：

1. sess: Tensorflow Session 对象，其中网络构图已经载入Tensorflow
2. logits: 模型输出
3. data_x: 训练样本
4. data_y: 训练标签

NAS工程需要方法返回下列结果，方便我们进行网络竞赛淘汰：

1. target: 网络的评估值，必须是浮点数
2. saver: Tensorflow Saver 对象，保存训练模型
3. log: 运行日志 (可以在memory/evaluator_log.txt中找到)

> [evaluator.py](../evaluator.py) 已经给出图像识别任务的评估方法。
> [evaluator_user.py](./../evaluator_user.py) 给出这个方法的模版。更详细的方法说明，可以参考下方[评估函数](user.md###评估函数)说明。

## 用户相关参数与模块函数

### 用户参数总览

+ nas_main 总控参数
  + num_gpu 运行环境GPU个数
  + block_num 堆叠网络块数量
  + add_data_per_round 每一轮竞赛增加数据大小
  + add_data_for_winner 竞赛胜利者的训练数据集大小(-1代表使用全部数据)
  + repeat_search 模块重复次数
+ enum 穷举模块参数
  + depth 枚举的网络结构的深度
  + width 枚举的网络结构的支链个数
  + max_depth 约束支链上节点的最大个数
+ eva
  + dataset_path 数据集路径
+ spl 采样参数
  + skip_max_dist 最大跨层长度
  + skip_max_num 最大跨层个数
  + space 搜索空间

### 数据集 Dataset

+ Dataset.\_\_init\_\_ 
    > 1. self.IMAGE_SIZE 图片尺寸
    > 2. self.NUM_CLASSES 分类数量/输出张量的最后一维大小
    > 3. self.NUM_EXAMPLES_FOR_TRAIN 训练数据集大小
    > 4. self.NUM_EXAMPLES_FOR_EVAL 评估数据集大小
    > 5. self.data_path = "./data" 数据集存放路径
+ Dataset.input 
    > **功能**： 将数据读入内存
    >
    > **Returns**:
    > 1. train_data 训练样本
    > 2. train_label 训练标签
    > 3. valid_data 评估样本
    > 4. valid_label 评估标签
    > 5. test_data 测试样本
    > 6. test_label 测试标签
+ Dataset.process \[*Options*\] 
    > **功能**： 对数据进行类似于图片增强等的处理
    >
    > **Args**:
    > 1. x 需要处理的数据
    >
    > **Returns**:
    > 1. x 处理完成的数据

### 评估函数

> 面对具体问题，自定义合适的评估算法。
> 网络结构和操作配置已经载入 Tensorflow，计算得到
> 评估值返回给 NAS，以此作为后续 NAS 淘汰竞赛的指标。

+ Evaluator.\_\_init\_\_ 
    > 1. self.batch_size batch大小
    > 2. self.model_path 模型存储的路径

+ Evaluator.\_make_layer 
    > **功能**： Method for constructing and calculating cell in tensorflow. Please notice that this function do not need rewrite,
    simply add the corresponding code can full fill its propose. See the instruction in the evaluator_user.py for specify use.
    >
    > **Args**:
    > 1. inputs: the input tensor of this operation
    > 2. cell: Class Cell(), hyper parameters for building this layer
    > 3. node: int, the index of this operation
    > 4. train_flag: boolean, indicating whether this is a training process or not
    >
    > **Returns**:
    > 1. layer: tensor.

+ Evaluator._eval
    > **功能**： The actual training process, including the definination of loss and train optimizer. 定义loss的计算和train opt的类型后启动sess.run()进行训练和评估，并返回对应的优化目标。这里的优化目标可以是单目标，如准确率；也可以是多目标，如综合考虑准确率和模型大小。
    >
    > **Args**:
    > 1. sess: Tensorflow Session
    > 2. logits: output tensor of the model, 2-D tensor of shape [self.batch_size, self.NUM_CLASS]
    > 3. data_x: input tensor
    > 4. data_y: input label, 2-D tensor of shape [self.batch_size, self.NUM_CLASS]
    > 5. *args (用户自定义参数)
    > 6. **kwg (用户自定义参数)
    >
    > **Returns**:
    > 1. target (float, 0 ~ 1): the optimization target, could be the accuracy or the combination of both time and accuracy, etc
    > 2. saver: Tensorflow Saver class
    > 3. log (string): log to be write and saved
