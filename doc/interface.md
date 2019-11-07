
# Interface

------------------------------

## Network

> No Method, Data only.

+ id (int, any)
+ graph_template (2d int list, adjacency list)
+ item_list (1d NetworkItem list)

## NetworkItem

> No Method, Data only.

+ id (int, any)
+ graph (2d int list, adjacency list)
+ cell_list (1d Cell list)
+ code (1d int list, depending on dimension)
+ score (int, 0 ~ 1)

## Cell (inherit from Tuple)

> No Method, Data only.
> ex. cell = Cell('conv', 48, 7, 'relu')

+ type (string, 'conv' or 'pooling')
+ 'conv' items
    1. filter_size (int, 1 ~ 1024)
    2. kernel_size (int, odd and 1 ~ 9)
    3. activation (string, valid values: )
        1. relu
        2. tanh
        3. sigmoid
        4. identity
        5. "leakyrelu"
+ 'pooling' items
    1. ptype (string, 'avg' or 'max' or 'global')
    2. kernel_size (int, 1 ~ 10)

## Nas

> The core of NAS Project.

### Config

+ PS_HOST (string, host address)
+ JOB_NAME (string, 'ps' or 'worker')
+ NUM_OPT_BEST (int, >= 1)
+ BLOCK_NUM (int, >= 1)
+ NUM_GPU (int, >= 1)
+ FINETUNE_THRESHOLD (int, ?) <!--TODO-->
+ SPL_NETWORK_ROUND (int, >= 1)

### Method

+ run
    > **Args**
    > 1. proc_pool (mutiprocessor.Pool)
    > **Returns**
    > 1. best_nn (Network)

## Enumerater

### Config

<!-- TODO -->
1. MAX_DEPTH (int, any)
2. MAX_WIDTH (int, any)
3. MAX_BRANCH_DEPTH (int, any)
4. ENUM_LOG_PATH (string, file path)

### Method

1. enumrate
    > **Args**: None
    > **Returns**
    > 1. pool (Network list)

## Evaluator

### Config

> Note: The range of IMAGE_SIZE, NUM_CLASSES, NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN, NUM_EXAMPLES_PER_EPOCH_FOR_EVAL depend on dateset.

+ IMAGE_SIZE (int, unknown)
+ NUM_CLASSES (int, unknown)
+ NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN (int, unknown)
+ NUM_EXAMPLES_PER_EPOCH_FOR_EVAL (int, unknown)
+ INITIAL_LEARNING_RATE (float, 1.0 ~ 1e-5)
+ MOVING_AVERAGE_DECAY (float, 1.0 ~ 1e-5)
+ REGULARAZTION_RATE (float, 1.0 ~ 1e-5)
+ BATCH_SIZE (int, <= 200)
+ EPOCH (int, any)
+ WEIGHT_DECAY (float, 0 ~ 1.0)
+ MOMENTUM_RATE (float, 0 ~ 1.0)
+ EVA_LOG_PATH (string, file path)
+ MODEL_SAVE_PATH (string, file path)
  
### Method

+ evaluate
    > **Args**:
    > 1. *graph_full*
    > 2. *cell_list*
    > 3. *pre_block*
    > 4. *is_bestNN*
    > 5. *update_pre_weight*
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
    > 1. *add_num* (int, *BATCH_SIZE* ~ *NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN* - *self.train_num*)
    >
    > **Returns**: None
  
## Sample

### Config

+ SKIP_MAX_DIST (int, 0 ~ MAX_DEPTH)
+ SKIP_MAX_NUM (int, 0 ~ MAX_DEPTH - 1)
+ CONV_SPACE (dict)
  + filter_size (1d int list, value as Cell.filter_size)
  + kernel_size (2d int list, value as Cell.kernel_size)
  + activation (1d string list, value as Cell.activation)
+ POOL_SPACE (dict)
  + pooling_type (1d string list, value as Cell.pooling_type)
  + kernel_size (1d int list, value as Cell.kernel_size)
+ POOL_SWITCH (boolean)
+ SPL_LOG_PATH

### Method

+ sample
    > **Args**:
    > 1. graph_part (2d int list, as NetworkUnit.graph_part)
    > 2. block_id (int, 0 ~ BLOCK_NUM - 1)
    >
    > **Returns**:
    > 1. cell: (class Cell list)
    > 2. graph: (2d int list, as NetworkUnit.graph_part)
+ update_model
    > **Args**:
    > 1. table (1d int list, depending on dimension)
    > 2. score （float, 0 ~ 1.0)
    >
    > **Returns**: None

## Predictor
<!-- TODO -->
### Method

+ predict
    > **Args**
    > 1. 
    >
    > **Returns**
    > 1. ops
    >