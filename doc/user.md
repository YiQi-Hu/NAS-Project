# User Cookbook

-------------------------

## 数据集 Dataset

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
+ Dataset.process \[Options\] 
    > **功能**： 对数据进行类似于图片增强等的处理
    >
    > **Args**:
    > 1. x 需要处理的数据
    >
    > **Returns**:
    > 1. x 处理完成的数据

## 评估函数

> 面对具体问题，自定义合适的评估算法。
> 网络结构和操作配置已经载入 Tensorflow，计算得到
> 评估值返回给 NAS，以此作为后续 NAS 淘汰竞赛的指标。
> evaluator_user.py <!-- TODO -->

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
+ utils.Datasize.control <!-- TODO -->

## 用户相关参数

> nas_config.py 用户相关的参数

+ nas_main 总控参数
  + num_gpu 运行环境GPU个数
  + block_num 堆叠网络块数量
  + add_data_per_round 每一轮竞赛增加数据大小
  + add_data_for_winner 竞赛胜利者的训练数据集大小(-1代表使用全部数据)
  + repeat_search
+ enum 穷举模块参数
  + depth
  + width
  + max_depth
+ spl 采样参数
  + skip_max_dist
  + skip_max_num

## 配置参数搜索空间与配置参数生成模板

如果想要自定义特定的网络节点操作配置，请完成以下几点：

1. 修改 class _Cell_ 函数
    1. \_\_getstate\_\_, \_\_setstate\_\_ 进程传递编码相关，具体要求请参考 pickle 官方文档。
    2. 可以放弃 _Cell_ 类型的语法糖，采用较为简单的设计。
2. 修改 NAS_CONFIG\['spl'\]\['space'\] 搜索空间
3. 修改 Evaluator._make_layer 转换成具体网络节点，并且实现自定义操作的函数

## NAS日志

1. 日志文件夹 .\memory
    1. 评估: evaluator_log.txt
    2. 子进程: subproc_log.txt
    3. 网络信息: network_info.txt
    4. 总控: nas_log.txt
2. 日志纪录对象Logger:
    > 提供全工程统一的日志纪录接口
    > 如果需要纪录更多更复杂的日志，可以重新实现Logger
    > 具体使用说明，请参考interface.md
