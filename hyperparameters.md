# Hyperparameters used in the PowerSGD paper

## Table 1: Rank-based compression

### Shared parameters

```
average_reset_epoch_interval: 30
distributed_backend: nccl
fix_conv_weight_norm: false
log_verbosity: 1
n_workers: 16
num_epochs: 300
num_train_tracking_batches: 1
optimizer_batch_size: 128
optimizer_decay_at_epochs: [150, 250]
optimizer_decay_with_factor: 10.0
optimizer_mom_before_reduce: false
optimizer_momentum_type: nesterov
optimizer_momentum: 0.9
optimizer_scale_lr_with_factor: 16
optimizer_scale_lr_with_warmup_epochs: 5
optimizer_wd_before_reduce: false
optimizer_weight_decay_bn: 0.0
optimizer_weight_decay_conv: 0.0001
optimizer_weight_decay_other: 0.0001
seed: 1, 2 and 3
task_architecture: ResNet18
task: Cifar
```

### SGD

```
optimizer_reducer: ExactReducer
optimizer_conv_learning_rate: 0.1
optimizer_learning_rate: 0.1
optimizer_memory: false
```

### Rank-R PowerSGD

```
optimizer_reducer: RankKReducer
optimizer_conv_learning_rate: 0.1
optimizer_learning_rate: 0.1
optimizer_memory: true
optimizer_reducer_rank: R
optimizer_reducer_reuse_query: true
```

### Unbiased Rank R

```
optimizer_reducer: HalfRankKReducer
optimizer_conv_learning_rate: 0.0125
optimizer_learning_rate: 0.0125
optimizer_memory: false
optimizer_reducer_rank: R
```

## Table 2: Warm-start

### Shared parameters

```
average_reset_epoch_interval: 30
checkpoints: []
distributed_backend: nccl
fix_conv_weight_norm: false
log_verbosity: 1
n_workers: 16
num_epochs: 300
num_train_tracking_batches: 1
optimizer_batch_size: 128
optimizer_conv_learning_rate: 0.1
optimizer_decay_at_epochs: [150, 250]
optimizer_decay_with_factor: 10.0
optimizer_learning_rate: 0.1
optimizer_memory: true
optimizer_mom_before_reduce: false
optimizer_momentum_type: nesterov
optimizer_momentum: 0.9
optimizer_reducer_n_power_iterations: 0
optimizer_reducer_rank: 2
optimizer_scale_lr_with_factor: 16
optimizer_scale_lr_with_warmup_epochs: 5
optimizer_wd_before_reduce: false
optimizer_weight_decay_bn: 0.0
optimizer_weight_decay_conv: 0.0001
optimizer_weight_decay_other: 0.0001
seed: 1, 2 and 3
task_architecture: ResNet18
task: Cifar
```

### Best approximation

```
optimizer_reducer: RankKReducer
optimizer_reducer_reuse_query: false
optimizer_reducer_n_power_iterations: 2
```

Note: this code path is not available anymore. We found that the SGD trajectory did not improve by adding more than 2 power-iteration steps.

### Warm start

```
optimizer_reducer: RankKReducer
optimizer_reducer_reuse_query: true
optimizer_reducer_n_power_iterations: 0
```

### Without warm start

```
optimizer_reducer: RankKReducer
optimizer_reducer_reuse_query: false
optimizer_reducer_n_power_iterations: 0
```

## Table 3: PowerSGD with varying rank

### Image classification - shared parameters

```
average_reset_epoch_interval: 30
distributed_backend: nccl
fix_conv_weight_norm: false
log_verbosity: 1
n_workers: 16
num_epochs: 300
num_train_tracking_batches: 1
optimizer_batch_size: 128
optimizer_conv_learning_rate: 0.1
optimizer_decay_at_epochs: [150, 250]
optimizer_decay_with_factor: 10.0
optimizer_learning_rate: 0.1
optimizer_mom_before_reduce: false
optimizer_momentum_type: nesterov
optimizer_momentum: 0.9
optimizer_scale_lr_with_factor: 16
optimizer_scale_lr_with_warmup_epochs: 5
optimizer_wd_before_reduce: false
optimizer_weight_decay_bn: 0.0
optimizer_weight_decay_conv: 0.0001
optimizer_weight_decay_other: 0.0001
seed: 1, 2 and 3
task_architecture: ResNet18
task: Cifar
```

### Image classification - SGD

```
optimizer_reducer: ExactReducer
optimizer_memory: false
```

### Image classification - Rank R

```
optimizer_memory: true
optimizer_reducer: RankKReducer
optimizer_reducer_rank: R
optimizer_reducer_reuse_query: true
```

### Language modeling - Shared parameters

```
average_reset_epoch_interval: 30
checkpoints: []
distributed_backend: nccl
fix_conv_weight_norm: false
log_verbosity: 1
n_workers: 16
num_epochs: 90
num_train_tracking_batches: 1
optimizer_batch_size: 64
optimizer_conv_learning_rate: 1.25
optimizer_decay_at_epochs: [60, 80]
optimizer_decay_with_factor: 10.0
optimizer_learning_rate: 1.25
optimizer_mom_before_reduce: false
optimizer_momentum_type: nesterov
optimizer_momentum: 0.0
optimizer_scale_lr_with_factor: 16
optimizer_scale_lr_with_warmup_epochs: 5
optimizer_wd_before_reduce: false
optimizer_weight_decay_bn: 0.0
optimizer_weight_decay_conv: 0.0
optimizer_weight_decay_other: 0.0
rank: 0
seed: 1, 2 and 3
task_architecture: ResNet18
task: LanguageModeling
```

### Language modeling - SGD

```
optimizer_reducer: ExactReducer
optimizer_memory: false
```

### Language modeling - Rank R

```
optimizer_reducer: RankKReducer
optimizer_memory: true
optimizer_reducer_rank: R
optimizer_reducer_reuse_query: true
```

## Table 4: Comparison of compressors for Error Feedback

### Shared parameters

```
average_reset_epoch_interval: 30
distributed_backend: nccl
fix_conv_weight_norm: false
log_verbosity: 1
n_workers: 16
num_epochs: 300
num_train_tracking_batches: 1
optimizer_batch_size: 128
optimizer_conv_learning_rate: 0.1
optimizer_decay_at_epochs: [150, 250]
optimizer_decay_with_factor: 10.0
optimizer_learning_rate: 0.1
optimizer_mom_before_reduce: false
optimizer_momentum_type: nesterov
optimizer_momentum: 0.9
optimizer_scale_lr_with_factor: 16
optimizer_scale_lr_with_warmup_epochs: 5
optimizer_wd_before_reduce: false
optimizer_weight_decay_bn: 0.0
optimizer_weight_decay_conv: 0.0001
optimizer_weight_decay_other: 0.0001
seed: 1, 2 and 3
task_architecture: ResNet18
task: Cifar
```

### No compression (SGD)

```
optimizer_reducer: ExactReducer
optimizer_memory: false
```

### Medium - Rank-7 PowerSGD

```
optimizer_memory: true
optimizer_reducer: RankKReducer
optimizer_reducer_rank: 7
optimizer_reducer_reuse_query: true
```

### Medium - Random Block

```
optimizer_memory: true
optimizer_reducer: RandomSparseBlockReducer
optimizer_reducer_rank: 7
```

### Medium - Random K

```
optimizer_memory: true
optimizer_reducer: RandomSparseReducer
optimizer_reducer_rank: 7
```

### Medium - Sign+Norm

```
optimizer_memory: true
optimizer_reducer: SignAndNormReducer
```

### Medium - Top K

```
optimizer_memory: true
optimizer_reducer: TopKReducer
optimizer_reducer_compression: 0.007352941176470588
```

### High - Rank-2 PowerSGD

```
optimizer_memory: true
optimizer_reducer: RankKReducer
optimizer_reducer_rank: 2
optimizer_reducer_reuse_query: true
```

### High - Random Block

```
optimizer_memory: true
optimizer_reducer: RandomSparseBlockReducer
optimizer_reducer_rank: 2
```

### High - Top K

```
optimizer_memory: true
optimizer_reducer: TopKReducer
optimizer_reducer_compression: 0.03125
```

## Table 5: Timing breakdown

### Shared parameters

```
average_reset_epoch_interval: 30
distributed_backend: nccl
fix_conv_weight_norm: false
num_epochs: 10
checkpoints: []
num_train_tracking_batches: 1
optimizer_batch_size: 128
optimizer_conv_learning_rate: 0.1
optimizer_decay_at_epochs: [150, 250]
optimizer_decay_with_factor: 10.0
optimizer_learning_rate: 0.1
optimizer_memory: false
optimizer_momentum_type: nesterov
optimizer_momentum: 0.9
optimizer_scale_lr_with_factor: 4
optimizer_scale_lr_with_warmup_epochs: 5
optimizer_mom_before_reduce: false
optimizer_wd_before_reduce: false
optimizer_weight_decay_conv: 0.0001
optimizer_weight_decay_other: 0.0001
optimizer_weight_decay_bn: 0.0
task: Cifar
task_architecture: ResNet18
seed: multiple
```

### Vary

```
log_verbosity: 1, 2
optimizer_reducer: ExactReducer, SignSGDwithMajorityVoteReducer, RankKReducer,
n_workers: 1, 2, 4, 8, 16
```

## Figure 3: Scaling

### Shared parameters

```
average_reset_epoch_interval: 30
checkpoints: []
distributed_backend: nccl
fix_conv_weight_norm: false
log_verbosity: 1
num_epochs: 10
num_train_tracking_batches: 1
optimizer_batch_size: 128
optimizer_conv_learning_rate: 0.1
optimizer_decay_at_epochs: [150, 250]
optimizer_decay_with_factor: 10.0
optimizer_learning_rate: 0.1
optimizer_memory: false
optimizer_mom_before_reduce: false
optimizer_momentum_type: nesterov
optimizer_momentum: 0.9
optimizer_scale_lr_with_factor: 4
optimizer_scale_lr_with_warmup_epochs: 5
optimizer_wd_before_reduce: false
optimizer_weight_decay_bn: 0.0
optimizer_weight_decay_conv: 0.0001
optimizer_weight_decay_other: 0.0001
seed: multiple
task_architecture: ResNet18
task: Cifar
```

### Vary

```
backend: nccl, gloo
optimizer_reducer: ExactReducer, SignSGDwithMajorityVoteReducer, RankKReducer,
n_workers: 1, 2, 4, 8, 16
optimizer_memory: true for RankKReducer
optimizer_mom_before_reduce: true for SignSGDwithMajorityVoteReducer
optimizer_momentum_type: exponential_moving_average for SignSGDwithMajorityVoteReducer

```

## Table 6: Comparison against other algorithms - image classification

### Shared parameters

```
average_reset_epoch_interval: 30
distributed_backend: nccl
fix_conv_weight_norm: false
log_verbosity: 1
n_workers: 16
num_epochs: 300
num_train_tracking_batches: 1
optimizer_batch_size: 128
optimizer_decay_at_epochs: [150, 250]
optimizer_decay_with_factor: 10.0
optimizer_mom_before_reduce: false
optimizer_momentum: 0.9
optimizer_scale_lr_with_factor: 16
optimizer_scale_lr_with_warmup_epochs: 5
optimizer_wd_before_reduce: false
optimizer_weight_decay_bn: 0.0
optimizer_weight_decay_conv: 0.0001
optimizer_weight_decay_other: 0.0001
seed: 1, 2 and 3
task_architecture: ResNet18
task: Cifar
```

### SGD

```
optimizer_reducer: ExactReducer
optimizer_conv_learning_rate: 0.1
optimizer_learning_rate: 0.1
optimizer_memory: false
optimizer_momentum_type: nesterov
optimizer_reducer_n_power_iterations: n.a.
optimizer_reducer_rank: n.a.
optimizer_mom_before_reduce: false

```

### Rank-2 PowerSGD

```
optimizer_reducer: RankRReducer
optimizer_conv_learning_rate: 0.1
optimizer_learning_rate: 0.1
optimizer_memory: true
optimizer_momentum_type: nesterov
optimizer_reducer_rank: 2
optimizer_mom_before_reduce: false
optimizer_reducer_reuse_query: true

```

### Atomo (rank 2)

```
optimizer_reducer: AtomoReducer
optimizer_conv_learning_rate: 0.1
optimizer_learning_rate: 0.1
optimizer_memory: false
optimizer_momentum_type: nesterov
optimizer_reducer_rank: 2
optimizer_mom_before_reduce: false

```

### Signum

```
optimizer_reducer: SignSGDwithMajorityVoteReducer
optimizer_conv_learning_rate: 0.00005
optimizer_learning_rate: 0.00005
optimizer_memory: false
optimizer_momentum_type: exponential_moving_average
optimizer_reducer_rank: 2
optimizer_mom_before_reduce: true
```

## Table 7: Comparison against others - language modeling

### Shared parameters

```
average_reset_epoch_interval: 30
checkpoints: []
distributed_backend: nccl
fix_conv_weight_norm: false
log_verbosity: 1
n_workers: 16
num_epochs: 90
num_train_tracking_batches: 1
optimizer_batch_size: 64
optimizer_conv_learning_rate: 1.25
optimizer_decay_at_epochs: [60, 80]
optimizer_decay_with_factor: 10.0
optimizer_learning_rate: 1.25
optimizer_momentum_type: nesterov
optimizer_scale_lr_with_factor: 16
optimizer_scale_lr_with_warmup_epochs: 5
optimizer_wd_before_reduce: false
optimizer_weight_decay_bn: 0.0
optimizer_weight_decay_conv: 0.0
optimizer_weight_decay_other: 0.0
rank: 0
seed: 1, 2 and 3
task_architecture: ResNet18
task: LanguageModeling
```

### SGD

```
optimizer_learning_rate: 1.25
optimizer_conv_learning_rate: 1.25
optimizer_memory: false
optimizer_mom_before_reduce: false
optimizer_momentum: 0.0
optimizer_momentum_type: nesterov
optimizer_reducer: ExactReducer
```

### Signum

```
optimizer_learning_rate: 0.00001
optimizer_conv_learning_rate: 0.00001
optimizer_memory: false
optimizer_mom_before_reduce: true
optimizer_momentum: 0.0
optimizer_momentum_type: exponential_moving_average
optimizer_reducer: SignSGDwithMajorityVoteReducer
```

### Rank-4 PowerSGD

```
optimizer_learning_rate: 1.25
optimizer_conv_learning_rate: 1.25
optimizer_memory: true
optimizer_mom_before_reduce: false
optimizer_momentum: 0.0
optimizer_momentum_type: nesterov
optimizer_reducer: RankKReducer
optimizer_reducer_rank: 4
optimizer_reducer_reuse_query: true
```
