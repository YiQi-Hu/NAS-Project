# User Manual

-------------------------
如何客制化 NAS 工程。

## 评估函数

> 面对具体问题，自定义合适的评估算法。
> 网络结构和操作配置已经载入 Tensorflow，计算得到
> 评估值返回给 NAS，以此作为后续 NAS 淘汰竞赛的指标。

+ Evaluator._eval
    > **Args**:
    > 1. sess
    > 2. user_pkg (tuple, value:)
    >    1. logits <!-- TODO -->
    >    2. data_x <!-- TODO -->
    >    3. data_y <!-- TODO -->
    >    4. train_flag <!-- TODO -->
    > 3. retrain (bool) *deprecated*
    >
    > **Returns**:
    > 1. precision (float, 0 ~ 1)

### 用户相关参数

> nas_config.py 用户相关的参数
> 具体含义请参考 interface.md

+ nas_main
  + num_gpu
  + block_num
+ enum
  + depth
  + width
  + max_depth
+ eva
  + task_name
  + image_size
  + num_classes
  + num_examples_for_train
  + num_examples_per_epoch_for_eval
  + initial_learning_rate
  + num_epochs_per_decay
  + learning_rate_decay_factor
  + moving_average_decay
  + batch_size
  + weight_decay
  + momentum_rate
  + model_path
  + dataset_path
  + eva_log_path
  + retrain_switch
  + learning_rate_type
  + boundaries
  + learing_rate
+ spl
  + skip_max_dist
  + skip_max_num

## 配置參數搜索空间与配置参数生成模板

> 如果想要自定义特定的网络节点操作配置，请完成以下几点：
>
> 1. 修改 class _Cell_ 函数：
>    1. \_\_getstate\_\_, \_\_setstate\_\_ 进程传递编码相关，具体要求请参考 pickle 官方文档。
>    2. \_check_valid 参数检查
>    3. _Cell_ 类型继承自元组，也可以放弃语法糖和严格参数检查，采用较为简单的设计。
> 2. 修改 NAS_CONFIG\['spl'\] 搜索空间
> 3. 修改 Evaluator._make_layer 轉換成具體網絡節點
>