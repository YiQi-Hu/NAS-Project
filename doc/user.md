# User Manual

-------------------------
如何客制化 NAS 工程。

## 数据集 Dataset

+ Dataset.\_\_init\_\_ <!-- TODO -->
+ Dataset.input <!-- TODO -->
+ Dataset.process \[Options\] <!-- TODO -->


## 评估函数

> 面对具体问题，自定义合适的评估算法。
> 网络结构和操作配置已经载入 Tensorflow，计算得到
> 评估值返回给 NAS，以此作为后续 NAS 淘汰竞赛的指标。
> evaluator_user.py <!-- TODO -->

+ Evaluator.\_\_init\_\_ <!-- TODO -->
+ Evaluator.\_make_layer <!-- TODO -->
+ Evaluator.add_data <!-- TODO -->
+ Evaluator._eval <!-- TODO -->
    > **Args**:
    > 1. sess
    > 2. logits
    > 3. data_x
    > 4. data_y
    > 5. *args (用户自定义参数)
    > 6. **kwg (用户自定义参数)
    >
    > **Returns**:
    > 1. precision (float, 0 ~ 1)
    > 2. saver (tf.Saver)
    > 3. log (string)

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
    2. (Options) \_check_valid 参数检查
    3. _Cell_ 类型继承自元组，也可以放弃语法糖和严格参数检查，采用较为简单的设计。
2. 修改 NAS_CONFIG\['spl'\] 搜索空间
3. 修改 Evaluator._make_layer 转换成具体网络节点，并且实现自定义操作的函数
4. Sampler 搜索空间构成 <!-- TODO -->

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
