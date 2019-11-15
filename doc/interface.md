
# Interface

------------------------------

## Network

> No Method, Data only.

+ id (int, any)
+ graph_template (2d int list, adjacency list)
+ item_list (1d NetworkItem list)
+ pre_block (1d NetworkItem list)
+ spl (class Sampler)

## NetworkItem

> No Method, Data only.

+ id (int, any)
+ graph (2d int list, adjacency list)
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

+ type (string, 'conv' or 'pooling')
+ 'conv' items
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

## info_str

> Except nas_config, every property else is string.

### nas_config

> Defined in *nas_config.json*.
>
> Please use `from info_str import nas_config` to get
> project's configuration.
>
> The following keys correspond to modul parameters:
>
> 1. enum -> Enumerater
> 2. eva -> Evalutor
> 3. spl -> Sampler
> 4. pred -> Predictor
>
> You can get Nas parameter directly with its name.
>
> Ex. `nas_config['num_gpu']`, `nas_config['enum']['max_depth']`

### Properties (string)

<!-- TODO -->

## Nas

> The core of nas Project.

### Config

+ num_opt_best (int, >= 1)
+ block_num (int, >= 1)
+ num_gpu (int, >= 1)
+ finetune_threshold (int, ?) <!--TODO-->
+ spl_network_round (int, >= 1)

### Method

+ run
    > **Args**:
    > 1. *proc_pool* (mutiprocessor.Pool)
    >
    > **Returns**:
    > 1. *best_nn* (Network)

## Enumerater

### Config

<!-- TODO -->
1. max_depth (int, any)
2. max_width (int, any)
3. max_branch_depth (int, any)
4. enum_log_path (string, file path)

### Method

1. enumrate
    > **Args**: None
    >
    > **Returns**:
    > 1. *pool* (Network list)

## Evaluator

### Config

> Note: The range of image_size, num_classes, num_examples_per_epoch_for_train, num_examples_per_epoch_for_eval depend on dateset.

+ image_size (int, unknown)
+ num_classes (int, unknown)
+ num_examples_per_epoch_for_train (int, unknown)
+ num_examples_per_epoch_for_eval (int, unknown)
+ initial_learning_rate (float, 1.0 ~ 1e-5)
+ moving_average_decay (float, 1.0 ~ 1e-5)
+ regularaztion_rate (float, 1.0 ~ 1e-5)
+ batch_size (int, <= 200)
+ epoch (int, any)
+ weight_decay (float, 0 ~ 1.0)
+ momentum_rate (float, 0 ~ 1.0)
+ boundaries (1d int list, )
+ learning_rate_type (string, )
+ learning_rate (1d float list, )
+ eva_log_path (string, file path)
+ model_save_path (string, file path)
  
### Method

+ evaluate
    > **Args**:
    > 1. *network_item* (NetworkItem)
    > 2. *is_bestNN* (boolean)
    > 3. *update_pre_weight* (boolean)
    >
    > **Returns**:
    > 1. *Accuracy* (float, 0 ~ 1.0)
    >
    > **Invalid**:
    > 1. *pre_block* = [] & *update_pre_weight* != True
    > 2. *update_pre_weight* = True, but *is_bestNN* = True before.
    > 3. *is_bestNN* = True & *update_pre_weight* = True
+ add_data
    > **Args**:
    > 1. *add_num* (int, *batch_size* ~ *num_examples_per_epoch_for_train* - *self.train_num*)
    >
    > **Returns**: None
  
## Sample

### Config

+ skip_max_dist (int, 0 ~ max_depth)
+ skip_max_num (int, 0 ~ max_depth - 1)
+ conv_space (dict)
  + filter_size (1d int list, value as Cell.filter_size)
  + kernel_size (2d int list, value as Cell.kernel_size)
  + activation (1d string list, value as Cell.activation)
+ pool_space (dict)
  + pooling_type (1d string list, value as Cell.pooling_type)
  + kernel_size (1d int list, value as Cell.kernel_size)
+ pool_switch (boolean)
+ spl_log_path (string, file path)

### Method

+ __init__
    > **Args**:
    > 1. graph
    > 2. block_num
    > **Returns**: None
    >
+ sample
    > **Args**: None
    >
    > **Returns**:
    > 1. *cell*: (class Cell list)
    > 2. *graph*: (2d int list, as NetworkUnit.graph_part)
    > 3. *table*
+ update_model
    > **Args**:
    > 1. *table* (1d int list, depending on dimension)
    > 2. *score* （float, 0 ~ 1.0)
    >
    > **Returns**: None
+ ops2table
    > **Args**
    > 1. *ops*
    > **Retruns**
    > 1. *table*
    >
+ convert
    > **Args**:
    > 1. *table*
    >
    > **Returns**:
    > 1. *cell_list* (1d Cell list)
    > 2. *graph_full*

## Predictor

### Method

+ predict
    > **Args**:
    > 1. *graph_ful* (2d int list, adjacency table)
    > 2. *pre_block*
    >
    > **Returns**:
    > 1. *ops*

+ train_model
    > **Args**:
    > 1. *graph_full*
    > 2. *cell_list*
    > **Returns**: None
    >
