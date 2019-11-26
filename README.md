# PowerSGD

Practical Low-Rank Gradient Compression for Distributed Optimization

Abstract:
We study gradient compression methods to alleviate the communication bottleneck in data-parallel distributed optimization. Despite the significant attention received, current compression schemes either do not scale well or fail to achieve the target test accuracy. We propose a new low-rank gradient compressor based on power iteration that can i) compress gradients rapidly, ii) efficiently aggregate the compressed gradients using all-reduce, and iii) achieve test performance on par with SGD. The proposed algorithm is the only method evaluated that achieves consistent wall-clock speedups when benchmarked against regular SGD with an optimized communication backend. We demonstrate reduced training times for convolutional networks as well as LSTMs on common datasets.

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

# Reference

If you use this code, please cite the following [paper](https://arxiv.org/abs/1905.13727)

    @inproceedings{vkj2019powerSGD,
      author = {Vogels, Thijs and Karimireddy, Sai Praneeth and Jaggi, Martin},
      title = "{{PowerSGD}: Practical Low-Rank Gradient Compression for Distributed Optimization}",
      booktitle = {NeurIPS 2019 - Advances in Neural Information Processing Systems},
      year = 2019,
      url = {https://arxiv.org/abs/1905.13727}
    }
