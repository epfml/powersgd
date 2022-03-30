import math
from copy import deepcopy

import torch
import torch.distributed as dist


class DistributedSampler(torch.utils.data.distributed.Sampler):
    """
    This is a copy of torch.utils.data.distributed.DistributedSampler (28 March 2019)
    with the option to turn off adding extra samples to divide the work evenly.
    """

    def __init__(self, dataset, add_extra_samples=True):
        self._dataset = dataset
        if torch.distributed.is_available():
            self._num_replicas = torch.distributed.get_world_size()
            self._rank = torch.distributed.get_rank()
        else:
            self._num_replicas = 1
            self._rank = 0
        self._add_extra_samples = add_extra_samples
        self._epoch = 0

        if add_extra_samples:
            self._num_samples = int(math.ceil(len(self._dataset) * 1.0 / self._num_replicas))
            self._total_size = self._num_samples * self._num_replicas
        else:
            self._total_size = len(self._dataset)
            num_samples = self._total_size // self._num_replicas
            rest = self._total_size - num_samples * self._num_replicas
            if self._rank < rest:
                num_samples += 1
            self._num_samples = num_samples

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self._epoch)
        indices = torch.randperm(len(self._dataset), generator=g).tolist()

        # add extra samples to make it evenly divisible
        if self._add_extra_samples:
            indices += indices[: (self._total_size - len(indices))]
        assert len(indices) == self._total_size

        # subsample
        indices = indices[self._rank : self._total_size : self._num_replicas]
        assert len(indices) == self._num_samples

        # This wasn't there before, which seems a bug?
        # Is the user supposed to do this?
        self.set_epoch(self._epoch + 1)

        return iter(indices)

    def __len__(self):
        return self._num_samples

    def set_epoch(self, epoch):
        self._epoch = epoch
