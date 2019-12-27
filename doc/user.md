# User Cookbook

-------------------------

## 概览

**NAS** (Neural Architecture Search) 算法，一种用于自动搜索比人工设计表现更优异的神经网络的自动机器学习算法。

工程主要分为五个部份：

1. Enumerater: 穷举网络结构，产生多个备选网络。
2. Predictor: 根据人类的先验知识，初始化合适的网络节点配置。
3. Evaluator: 评估网络表现结果，用来判断网络结构的优劣。
4. Sampler, Optimizer: 节点配置采样和优化，提升网络的表现。

工程总控的总体搜索过程：

1. 初始化备选网络池 (Enumerater, Predictor)
2. 评估网络结构的表现 (Evaluator)
3. 根据评估结果，进行淘汰竞赛，从网络池中删去劣等的网络结构
4. 对于存活的网络，采样和优化网络节点配置 (Sampler, Optimizer)
5. 反复竞赛之后，最终将挑选出优胜的网络
6. 以优胜的网络为前置网络模块，接在下一轮竞赛的网络结构前面。
7. NAS算法得到以优胜的网络组成链状网络模块，作为最佳网络结构返回

## 运行 NAS 算法

创建 Nas 总控对象，run 方法最终返回最佳网络结果。

```python
from multiprocessing import Pool
from info_str import NAS_CONFIG

NUM_GPU = NAS_CONFIG['nas_main']["num_gpu"]
nas = Nas(Pool(processes=NUM_GPU))
best_nn = nas.run()
```

## 配置具体任务

为了在不同的具体任务上运行 Nas 算法，根据具体任务内容，需要补完
工程的代码模版，主要分为三个部份：

1. 修改数据集和评估参数
2. 实现网络评估方法
3. 配置搜索空间 (可选)

### 修改数据集和评估参数

根据具体任务，设置 `Evaluator.__init__` 参数：

1. *input_shape* 图片尺寸
2. *output_shape* 分类数量/输出张量的最后一维大小
3. *batch_size* 训练数据集大小
4. *train_data* 训练样本
5. *train_label* 训练标签
6. *valid_data* 评估样本
7. *valid_label* 评估标签
8. *model_path* 模型路径

> **注意**: 请勿更改下列参数 (NAS工程相关)
>
> 1. *os.environ\["TF_CPP_MIN_LOG_LEVEL"\]*
> 2. *train_num*
> 3. *block_num* 
> 4. *log*

### 实现网络评估方法

请实现 `Evalutor._eval` 方法。我们已经给出方法的模版，以及评估需要的参数：

1. _sess_: Tensorflow Session 对象，其中网络构图已经载入Tensorflow
2. _logits_: 模型输出 (tf.Tensor对象)
3. _data\_x_: 训练样本 (tf.placeholder对象)
4. _data\_y_: 训练标签 (tf.placeholder对象)

NAS工程需要方法返回下列结果，方便我们进行网络竞赛淘汰：

1. _target_: 网络的评估值，必须是浮点数
2. _saver_: tf.Saver对象，保存训练模型
3. _log_: 运行日志 (内容可以在memory/evaluator_log.txt中找到)

> [evaluator.py](../evaluator.py) 已经给出图像识别任务的评估方法。
> [evaluator_user.py](./../evaluator_user.py) 给出这个方法的模版。更详细的方法说明，可以参考下方[评估函数](user.md###评估函数)说明。

### 配置搜索空间

如果想要配置搜索空间，需要添加自定义特定的网络节点操作配置。请完成以下几点：

1. 修改 `NAS_CONFIG['spl']['space']` 搜索空间
2. 修改 `Evaluator._make_layer` 转换成具体网络节点，并且实现自定义操作的函数

#### 具体范例

打算加入新的操作 **Separable Convolution** 到神经网络，其中包含几个参数 **filter_size**, **kernel_size**, **activation**

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

> **注意**：当参数是**1维列表**，代表**全体网络节点**的参数搜索空间；当参数是**2维列表**，子表代表**单个网络模块**的参数搜索空间(按照列表次序)。

接下来，只需要简单地放入配置参数文件 [nas_config.json](../nas_config.json)/spl/space 底下

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

最后，在 `Evaluator._make_layer` ，给出配置参数的具体操作方法

```python
def _make_layer(self, inputs, cell, node):
    if cell.type == 'sep_conv':
        layer = self._makeconv(inputs, cell, node)
    else:
        assert False, "Wrong cell type!"
    return layer

def _makesep_conv(self, inputs, hplist, node):
    # TODO
    return sep_conv_layer
```

> **Separable Convolution** 已经在NAS工程中实现，请作为参考。

## 其他常见问题

### 如何控制 NAS 算法运行细节？

可以透过修改配置参数，让NAS工程可以在不同环境执行不同任务。

下列为用户相关的参数：

+ nas_main 总控参数
  + _num\_gpu_ 运行环境GPU个数
  + _block\_num_ 堆叠网络块数量
  + _add\_data\_per\_round_ 每一轮竞赛增加数据大小
  + _add\_data\_for\_winner_ 竞赛胜利者的训练数据集大小 (-1代表使用全部数据)
  + _repeat\_search_ 模块重复次数
+ enum 穷举模块参数
  + _depth_ 枚举的网络结构的深度
  + _width_ 枚举的网络结构的支链个数
  + _max\_depth_ 约束支链上节点的最大个数
+ spl 采样参数
  + _skip\_max\_dist_ 最大跨层长度
  + _skip\_max\_num_ 最大跨层个数

> 所有工程相关配置参数都定义于 [nas_config.py](../nas_config.py)
>
> 如果想知道更多参数细节，请参考 [interface.md](interface.md)

### 如何找到工程日志文件？

所有日志文件都放在工程文件夹 *memory：*

1. 评估过程 evaluator_log.txt
2. 子进程 subproc_log.txt
3. 网络信息 network_info.txt
4. 总控 nas_log.txt

> 日志纪录对象 **Logger** 提供全工程统一的日志纪录接口
> 如果需要纪录更多更复杂的日志，可以重新实现 **Logger**。
> 具体使用说明，请参考 [interface.md](interface.md)

## 用户相关参数与模块函数

### *用户参数总览*

+ nas_main 总控参数
  + _num\_gpu_ 运行环境GPU个数
  + _block\_num_ 堆叠网络块数量
  + _add\_data\_per\_round_ 每一轮竞赛增加数据大小
  + _add\_data\_for\_winner_ 竞赛胜利者的训练数据集大小(-1代表使用全部数据)
  + _repeat\_search_ 模块重复次数
+ enum 穷举模块参数
  + _depth_ 枚举的网络结构的深度
  + _width_ 枚举的网络结构的支链个数
  + _max\_depth_ 约束支链上节点的最大个数
+ eva
  + _dataset\_path_ 数据集路径
+ spl 采样参数
  + _skip\_max\_dist_ 最大跨层长度
  + _skip\_max\_num_ 最大跨层个数
  + _space_ 搜索空间

### 数据集

+ Dataset.\_\_init\_\_ 
    > **Args**:
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
