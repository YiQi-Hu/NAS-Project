
# Interface

------------------------------

## Network

> No Method, Data only.

+ id (int, any)
+ graph_part (2d int list, adjacency list)
+ item_list (1d NetworkItem list)
+ pre_block (1d NetworkItem list, class static variable)
+ spl (class Sampler)

## NetworkItem

> No Method, Data only.

+ id (int, any)
+ graph_full (2d int list, adjacency list)
+ cell_list (1d Cell list)
+ code (1d int list, depending on dimension)
+ score (int, 0 ~ 1)

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

+ log_dir (string, './NAS-PROJECT/memory')
+ evalog_path (string (string, log_dir + 'evaluator_log.txt')
+ subproc_log (string, log_dir + 'subproc_log.txt')
+ network_info_path (string, log_dir + 'network_info.txt')
+ naslog_path (string, log_dir + 'nas_log.txt')
+ MF_TEMP (3d string dict, moudle X function X ACTION -> logger template string)

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
> Ex. `nas_config['num_gpu']`, `nas_config['enum']['max_depth']`

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
> Example:
>     NAS_LOG = Logger() # 'Nas.run' func in nas.py
>     NAS_LOG << 'enuming'

+ _eva_log = (string, from ifs.evalog_path)
+ _sub_proc_log = (string, from ifs.subproc_log_path)
+ _network_log = (string, from ifs.network_info_path)
+ _nas_log = (string, from ifs.naslog_path)
+ _log_map (2d string dict, module x func -> log)

## Nas

> The core of nas Project.

### Config

+ num_opt_best (int, >= 1)
+ block_num (int, >= 1)
+ num_gpu (int, >= 1)
+ finetune_threshold (int, ?)
+ spl_network_round (int, >= 1)
+ eliminate_policy (str, "best")
+ pattern (string, "Global" or "Block")
+ add_data_per_round (int, > 0)
+ add_data_for_winner (int, > 0 or -1(all))
+ add_data_mode ?
+ init_data_size ?
+ data_increase_scale ?
+ add_data_for_confirm_train ?
+ repeat_search (int, >= 1)

### Method

+ run
    > **Args**:
    > 1. *proc_pool* (mutiprocessor.Pool)
    >
    > **Returns**:
    > 1. *best_nn* (Network)

## Enumerater

### Config

1. depth (int, any)
2. width (int, any)
3. max_depth (int, any)

### Method

1. enumrate
    > **Args**: None
    >
    > **Returns**:
    > 1. *pool* (1d Network list)

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
+ search_epoch (int, any, epochs when search)
+ retrain_epoch (int, epochs when retrain)
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

+ pool_switch ?
+ skip_max_dist (int, 0 ~ max_depth)
+ skip_max_num (int, 0 ~ max_depth - 1)
+ space (dict, user defined)

### Method

+ \_\_init\_\_
    > **Args**:
    > 1. graph_part (2d int list, as Network.graph_part)
    > 2. block_num (int, 0 ~ any)
    >
    > **Returns**: None
    >
+ sample
    > **Args**: None
    >
    > **Returns**:
    > 1. *cell*: (1d Cell list)
    > 2. *graph_full*: (2d int list, as NetworkItem.graph_full)
    > 3. *table*: (1d int list, depending on dimension)
+ update_model
    > **Args**:
    > 1. *table* (1d int list, depending on dimension)
    > 2. *score* （float, 0 ~ 1.0)
    >
    > **Returns**: None
+ ops2table
    > **Args**
    > 1. *ops*
    >
    > **Retruns**:
    > 1. *table*: (1d int list, depending on dimension)
    >
+ convert
    > **Args**:
    > 1. *table*: (1d int list, depending on dimension)
    >
    > **Returns**:
    > 1. *cell_list*: (1d Cell list)
    > 2. *graph_full*: (2d int list, as NetworkItem.graph_full)

## Predictor

### Method

+ predict
    > **Args**:
    > 1. *graph_ful* (2d int list, adjacency table)
    > 2. *pre_block* (1d NetworkItem list)
    >
    > **Returns**:
    > 1. *ops*

+ train_model
    > **Args**:
    > 1. *graph_full*: (2d int list, as NetworkItem.graph_full)
    > 2. *cell_list*: (1d Cell list)
    > **Returns**: None
    >

## Optimizer

### Config

+ sample_size (int, ?)
+ budget (int, ?)
+ positive_num (int, ?)
+ rand_probability (float, ?)
+ uncertain_bit (int, ?)
