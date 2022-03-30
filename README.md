# PowerSGD

Practical Low-Rank Gradient Compression for Distributed Optimization

[Video](https://www.youtube.com/watch?v=xVxSu7KGtHw)

Abstract:
We study gradient compression methods to alleviate the communication bottleneck in data-parallel distributed optimization. Despite the significant attention received, current compression schemes either do not scale well or fail to achieve the target test accuracy. We propose a new low-rank gradient compressor based on power iteration that can i) compress gradients rapidly, ii) efficiently aggregate the compressed gradients using all-reduce, and iii) achieve test performance on par with SGD. The proposed algorithm is the only method evaluated that achieves consistent wall-clock speedups when benchmarked against regular SGD with an optimized communication backend. We demonstrate reduced training times for convolutional networks as well as LSTMs on common datasets.


## Reference implementation

This is a reference implementation for the PowerSGD algorithm.

Installation:

```bash
pip install git+https://github.com/epfml/powersgd.git
```

Usage:

```diff
+ from powersgd import PowerSGD, Config, optimizer_step

  model = torchvision.models.resnet50(pretrained=True)
  params = list(model.parameters())
  optimizer = torch.optim.SGD(params, lr=0.1)

+ powersgd = PowerSGD(params, config=Config(
+     rank=1,  # lower rank => more aggressive compression
+     min_compression_rate=10,  # don't compress gradients with less compression
+     num_iters_per_step=2,  #   # lower number => more aggressive compression
+     start_compressing_after_num_steps=0,
+ ))

  for each batch:
      loss = ...
-     optimizer.zero_grad()
      loss.backward()
-     optimizer.step()
+     optimizer_step(optimizer, powersgd)
```

## PyTorch implementation
PyTorch features an implementation of PowerSGD as a [communucation hook](https://pytorch.org/docs/stable/ddp_comm_hooks.html) for `DistributedDataParallel` models.
Because of the integration with DDP, the code is more involved than the code in this repository.
## Research code

Research code for the experiments in the [PowerSGD paper](https://arxiv.org/abs/1905.13727) is located under [paper-code](./paper-code/README.md).

## Selected follow-up work 
- [(Ramesh et al., 2021 - DALL-E)](https://arxiv.org/abs/2102.12092) share valuable recommendations in using PowerSGD for large-scale transformer training.
- (Please submit a PR if you want to be included in this list.)


# Reference

If you use this code, please cite the following [paper](https://arxiv.org/abs/1905.13727)

    @inproceedings{vkj2019powersgd,
      author = {Vogels, Thijs and Karimireddy, Sai Praneeth and Jaggi, Martin},
      title = "{{PowerSGD}: Practical Low-Rank Gradient Compression for Distributed Optimization}",
      booktitle = {NeurIPS 2019 - Advances in Neural Information Processing Systems},
      year = 2019,
      url = {https://arxiv.org/abs/1905.13727}
    }
