# Code organization

### A few pointers

-   [train.py](train.py) is the entrypoint.
-   [gradient_reducers.py](gradient_reducers.py) implements communication algorithms.
-   [Core of the PowerSGD algorithm](gradient_reducers.py#L665)
-   Optimization problems can be found under [tasks/](tasks/__init__.py).
-   [Hyperparameters](hyperparameters.md) for the experiments in the [paper](https://arxiv.org/abs/1905.13727).

### Distributed training & changing config

```python
import train

# Configure the worker
train.config["n_workers"] = 4
train.config["rank"] = 0 # number of this worker in [0,4).

# Override some hyperparameters to train PowerSGD
train.config["optimizer_scale_lr_with_factor"] = 4  # workers
train.config["optimizer_reducer"] = "RankKReducer"
train.config["optimizer_reducer_rank"] = 4
train.config["optimizer_memory"] = True
train.config["optimizer_reducer_reuse_query"] = True
train.config["optimizer_reducer_n_power_iterations"] = 0

# You can customize the outputs of the training script by overriding these members
train.output_dir = "choose_a_directory"
train.log_info = your_function_pointer
train.log_metric = your_metric_function_pointer

# Start training
train.main()
```

Note that `torch.distributed` uses global state, so you cannot easily run `train.main()` multiple times after each other in the same script.
