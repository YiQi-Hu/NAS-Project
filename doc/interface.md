
# Interface

------------------------------

## Network

> No Method, Data only.

+ id (int, any) 枚举出的每一个网络的固定编号
+ graph_template (2d int list, adjacency list) 枚举出的拓扑结构，不含跨层连接和操作配置，以邻接表的形式呈现
+ item_list (1d NetworkItem list) 配置列表，每个元素为一条配置，每条配置相当于一个完整的网络
+ pre_block (1d NetworkItem list, class static variable) 已经固定下来的block对应的配置
+ spl (class Sampler) 该网络单独维护的采样方法

## NetworkItem

> No Method, Data only.

+ id (int, any) 一个网络中每条配置对应的固定编号
+ graph_full (2d int list, adjacency list) 完整的拓扑结构
+ cell_list (1d Cell list) 对应的操作配置
+ code (1d int list, depending on dimension) 该条配置在采样模块中呈现的编码
+ score (int, 0 ~ 1) 该条配置的评分，由评估模块给出

## Cell (inherit from Tuple)

> No Method, Data only.
>
> **Example:**
>
> ```python
> cell = Cell('conv', 48, 7, 'relu')
> cell.type # 'conv'
> cell.filter_size # 48
> cell # ('conv', 48, 7, 'relu')
>```

+ type (string, 'conv' or 'pooling' or 'sep_conv')
+ 'sep_conv', 'conv' items
    1. filter_size (int, 1 ~ 1024)
    2. kernel_size (int, odd and 1 ~ 9)
    3. activation (string, valid values: )
        1. relu
        2. tanh
        3. sigmoid
        4. identity
        5. leakyrelu
        6. relu6
+ 'pooling' items
    1. ptype (string, 'avg' or 'max' or 'global')
    2. kernel_size (int, 1 ~ 10)
+ Cell.get_format
    > **Args**:
    > 1. *cell_type* (string, type of cell defined in configuration)
    >
    > **Returns**:
    > 1. *format* (tuple, keys order)

## info_str (abbr. ifs)

> Except nas_config, every property else is string.

+ log_dir (string, './NAS-PROJECT/memory') 日志存储目录
+ evalog_path (string (string, log_dir + 'evaluator_log.txt') 评估日志
+ subproc_log (string, log_dir + 'subproc_log.txt') 子进程日志
+ network_info_path (string, log_dir + 'network_info.txt') 网络结构信息
+ naslog_path (string, log_dir + 'nas_log.txt') 总控搜索日志
+ MF_TEMP (3d string dict, moudle X function X ACTION -> logger template string) 日志模板索引

## nas_config

> Defined in *nas_config.json*.
>
> Please use `from info_str import nas_config` to get
> project's configuration.
>
> The following keys correspond to modul parameters:
>
> 1. nas_main, core -> Nas
> 2. enum -> Enumerater
> 3. eva -> Evalutor
> 4. opt -> Optimizer
> 5. spl -> Sampler
> 6. pred -> Predictor
>
> You can get Nas parameter directly with its name.
>
> Example:
>```python 
> NAS_CONFIG['num_gpu']
> NAS_CONFIG['enum']['max_depth']
>```

## Logger

> Wrtie log or print system information.
> The specified log templeate is defined in info_str.py
>
> Args:
>
> 1. args (string or tuple, non-empty)
>    When it's tuple, its value is string.
>         The first value must be action.
> Return:
>     None
>
> Example:
>
>```python
>     NAS_LOG = Logger() # 'Nas.run' func in nas.py
>     NAS_LOG << 'enuming'
>```
>

+ _eva_log = (string, from ifs.evalog_path)
+ _sub_proc_log = (string, from ifs.subproc_log_path) 
+ _network_log = (string, from ifs.network_info_path)
+ _nas_log = (string, from ifs.naslog_path)
+ _log_map (2d string dict, module x func -> log)

## Nas

> The core of nas Project.

### Config

+ num_opt_best (int, >= 1) 竞赛结束，就赢家的基础上继续进行采样评估的次数
+ block_num (int, >= 1) 搜索的block数量
+ num_gpu (int, >= 1) 使用的GPU数量
+ finetune_threshold (int, ?) 竞赛后期开始进行finetune的阶段，当竞赛者个数小于此设定值时，开始finetune
+ spl_network_round (int, >= 1) 每轮竞赛每个网络进行采样评估的次数
+ eliminate_policy (str, "best") 减半策略（按网路的最优评分或者综合评分）
+ pattern (string, "Global" or "Block") 搜索模式
+ add_data_per_round (int, > 0) 每轮竞赛添加数据量
+ add_data_mode (string, "linear" or "scale") 数据增长方式（线性还是指数）
+ init_data_size (int, > 0) 初始添加数据量
+ data_increase_scale (float, > 1) 数据以指数增长时，底数的大小
+ add_data_for_confirm_train (int, -1 or > 0) 固定当前block网络结构前的训练所添加的数据量
+ repeat_num (int, >= 1) 搜得block中的结构重复堆叠的次数

### Method

+ run
    > **Args**: None
    >
    > **Returns**:
    > 1. *Network.pre_block* (1d NetworkItem list, and its length equals to block_num) 搜得的最终网络结构
    > 2. *retrain_score* retrain的评分

## Enumerater

### Config

1. depth (int, any) 所枚举的网络结构的深度
2. width (int, any) 所枚举的网络结构的支链个数
3. max_depth (int, any) 约束支链上节点的最大个数

### Method

1. enumrate
    > **Args**: None
    >
    > **Returns**:
    > 1. *pool* (1d Network list) 返回由base中Network结构组成的list

## Evaluator

### Config

> Note: The range of image_size, num_classes, num_examples_per_epoch_for_train, num_examples_per_epoch_for_eval depend on dateset.

+ task_name (string, value:)
    1. cifar-10
    2. cifar-100
    3. imagnet
+ image_size (int, size of the input image, 2nd and 3rd dimension of the input tensor)
+ num_classes (int, for image classification task, the number of class, the last dimension of output tensor)
+ num_examples_for_train (int, the number of dataset used for train, apart from validation)
+ num_examples_for_eval (int, the number of dataset used for validation)
+ initial_learning_rate (float, 1.0 ~ 1e-5, the initial learning rate)
+ weight_decay (float, 0 ~ 1.0, L2 factor in loss function)
+ momentum_rate (float, 0 ~ 1.0, momentum rate when use momentum optimizer)
+ batch_size (int, <= 200, batch size, may cause OOM error when set too big)
+ model_path (string, file path of saved model)
+ dataset_path (string, file path of data set)

### Method

+ evaluate
    > **Args**:
    > 1. *network* (NetworkItem, the network to be evaluated)
    > 2. *pre_block* (1d list of NetworkItem, blocks precede this network )
    > 3. *is_bestNN* (boolean, indicator of whether this network needs to be saved or not)
    > 4. *update_pre_weight* (boolean, indicator of whether need to update the weight of previous block)
    >
    > **Returns**:
    > 1. *Score* (float, 0 ~ 1.0)
    >
    > **Invalid**:
    > 1. *pre_block* = [] & *update_pre_weight* != True
    > 2. *update_pre_weight* = True, but *is_bestNN* = True before.
    > 3. *is_bestNN* = True & *update_pre_weight* = True
+ retrain
    > **Args**:
    >
    > **Returns**:
    > 1. *Score* (float, 0 ~ 1.0)
+ set_data_size
    > **Args**:
    > 1. *num* (int, *batch_size* ~ *num_examples_per_epoch_for_train* - *self.train_num*)
    >
    > **Returns**: None
+ set_epoch
    > **Args**:
    > 1. *epoch* (int)
    >
    > **Returns**: None

## Sampler

### Config

+ pool_switch (int ,0 or 1) 控制搜索空间是否加入池化操作，block搜索下设置为0
+ skip_max_dist (int, 0 ~ max_depth) 最大跨层长度
+ skip_max_num (int, 0 ~ max_depth - 1) 最大跨层个数
+ space (dict, user defined) 搜索空间

### Method

+ \_\_init\_\_
    > Sample类初始化
    >
    > **Args**:
    > 1. graph_part (2d int list, as Network.graph_template) 类型二维列表，是Network类的graph_template参数
    > 2. block_num (int, 0 ~ any) 搜索空间block_num id 
    >
    > **Returns**: None
    >
+ sample
    > sample方法
    > 作用：进行一次采样
    > **Args**: None 传入参数空
    >
    > **Returns**: 返回值三个
    > 1. *cell*: (1d Cell list) 配置列表
    > 2. *graph_full*: (2d int list, as NetworkItem.graph) 完整的拓扑结构(包含跨层链接) 是NetworkItem类的graph参数
    > 3. *table*: (1d int list, depending on dimension) cell和graph_full 所对应的优化空间的code编码
+ update_model
    > 更新模型方法
    >
    > **Args**:
    > 1. *table* (1d int list, depending on dimension) 某一组拓扑结构和配置列表的code编码 根据优化空间定义的
    > 2. *score* （float, 0 ~ 1.0) 评估返回后的分数
    >
    > **Returns**: None
+ ops2table
    > 预测模块的结果转化为code编码的方法
    >
    > **Args**
    > 1. *ops* (2d list) 预测模块传入特定的ops参数
    >
    > **Retruns**:
    > 1. *table*: (1d int list, depending on dimension) 返回一组code编码，根据优化空间的定义
    >
+ convert
    > code转cell_list和graph方法
    >
    > **Args**:
    > 1. *table*: (1d int list, depending on dimension) code编码，根据优化空间的定义
    >
    > **Returns**:
    > 1. *cell_list*: (1d Cell list) 一组配置列表
    > 2. *graph_full*: (2d int list, as NetworkItem.graph) 一个完整的拓扑结构(包含跨层链接)

## Predictor

### Method

+ predict
    > **Args**:
    > 1. *graph_ful* (2d int list, adjacency table) #一个不加跨层连接的拓扑结构
    > 2. *pre_block* (1d NetworkItem list) #之前block的拓扑结构，是Network类的item_list
    >
    > **Returns**:
    > 1. *ops* #一个二维的序列，输入参数graph_full的操作配置，只包含卷积操作的大小和核个数，或是否为池化操作

+ train_model
    > **Args**:
    > 1. *graph_full*: (2d int list, as NetworkItem.graph_full) #一个网络完整的拓扑结构
    > 2. *cell_list*: (1d Cell list) #一组配置列表
    > **Returns**: None
    >

## Optimizer

### Config

+ sample_size (int, 优化过程中维护的标记为负的样本数量)
+ budget (int, 总的采样次数)
+ positive_num (int, 优化过程中维护的标记为正的样本数量)
+ rand_probability (float, 从学习得到的分类区域内采样的概率，一般设为接近于1的数)
+ uncertain_bit (int, 采样时随机的维度数，小表示鼓励利用，大表示鼓励探索)
