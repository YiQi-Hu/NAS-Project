# User Manual

-------------------------
如何客制化 NAS 工程。

## 數據集 Dataset

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
+ Evaluator._eval <!-- TODO -->
    > **Args**:
    > 1. sess
    > 2. logits
    > 3. data_x
    > 4. data_y
    > 5. *args (用戶自定義參數)
    > 6. **kwg (用戶自定義參數)
    >
    > **Returns**:
    > 1. precision (float, 0 ~ 1)
    > 2. saver (tf.Saver)
    > 3. log (string)

## 用户相关参数

> nas_config.py 用户相关的参数
> 具体含义请参考 interface.md

+ nas_main
  + num_gpu
  + block_num
  + num_examples_for_train
  + num_examples_per_epoch_for_eval
  + repeat_search
+ enum
  + depth
  + width
  + max_depth
+ spl
  + skip_max_dist
  + skip_max_num

## 配置參數搜索空间与配置参数生成模板

如果想要自定义特定的网络节点操作配置，请完成以下几点：

1. 修改 class _Cell_ 函数
    1. \_\_getstate\_\_, \_\_setstate\_\_ 进程传递编码相关，具体要求请参考 pickle 官方文档。
    2. (Options) \_check_valid 参数检查
    3. _Cell_ 类型继承自元组，也可以放弃语法糖和严格参数检查，采用较为简单的设计。
2. 修改 NAS_CONFIG\['spl'\] 搜索空间
3. 修改 Evaluator._make_layer 轉換成具體網絡節點，並且實現自定義操作的函數
4. Sampler 搜索空間構成 <!-- TODO -->

## NAS日誌對象 Logger

log_path