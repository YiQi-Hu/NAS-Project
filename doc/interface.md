
# Interface

------------------------------

## Nas

<!-- TODO -->
### Config

+ BLOCK_NUM

## NetworkUnit

<!-- TODO -->

### Properties
<!-- TODO -->
### Method

+ init_sample
+ sample
    > **Args**:
    > 1. *block_id* (int, 0 ~ BLOCK_NUM - 1)
    >
    > **Returns**: None

## Cell

<!-- TODO -->

## Enumerater

### Config

<!-- TODO -->
1. MAX_DEPTH

### Method

<!-- TODO -->

## Evaluator

### Properties

<!-- TODO -->

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
+ LOG_SAVE_PATH (string, file path)
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
+ GLOBAL_CONV (dict)
  + filter_size (1d int list, value as Cell.filter_size)
  + kernel_size (2d int list, value as Cell.kernel_size)
  + activation (1d string list, value as Cell.activation)
+ GLOBAL_POOL (dict)
  + pooling_type (1d string list, value as Cell.pooling_type)
  + kernel_size (1d int list, value as Cell.kernel_size)
+ BLOCK_CONV (dict)
  + file_size (**2d** int list, value as Cell.filter_size)
  + kernel_size (1d int list, value as Cell.kernel_size)
+ SPL_NETWORK_ROUND (int, >= 1)
+ SPL_PATTERN (string, "Global" or "Block")

### Method

+ sample
    > **Args**:
    > 1. graph_part (2d int list, as NetworkUnit.graph_part)
    > 2. block_id (int, 0 ~ BLOCK_NUM - 1)
    >
    > **Returns**:
    > 1. cell: (class Cell list)
    > 2. graph: (2d int list, as NetworkUnit.graph_part)
+ update_opt_model
    > **Args**:
    > 1. table (1d int list, depending on dimension)
    > 2. score （float, 0 ~ 1.0)
    >
    > **Returns**: None
+ init_p
    > **Args**:
    > 1. op (list, result from predictor)
    >
    > **Retruns**:
    > 1. table (int, depending on dimension)
