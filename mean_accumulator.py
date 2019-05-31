import torch
from copy import deepcopy


class MeanAccumulator:
    def __init__(self, update_weight=1):
        self.average = None
        self.counter = 0
        self.update_weight = update_weight

    def value(self):
        if isinstance(self.average, dict):
            return {k: v.value() for k, v in self.average.items()}
        elif isinstance(self.average, list):
            return [v.value() for v in self.average]
        else:
            return self.average

    def reduce(self):
        """Reduce over workers"""
        if not torch.distributed.is_available() or torch.distributed.get_world_size() == 1:
            # Skip this if there is only one worker
            return

        if isinstance(self.average, dict):
            for key in sorted(self.average.keys()):
                self.average[key].reduce()
        elif isinstance(self.average, list):
            for avg in self.average:
                avg.reduce()
        else:
            device = "cuda" if torch.distributed.get_backend() == "nccl" else "cpu"
            total_count = torch.tensor(self.counter, dtype=torch.float32, device=device)
            handle_tc = torch.distributed.all_reduce(total_count, async_op=True)

            # Average * count
            if isinstance(self.average, torch.Tensor):
                multiplied = self.average.clone()
            else:
                multiplied = torch.tensor(self.average, dtype=torch.float32, device=device)
            multiplied.mul_(self.counter)
            handle_mul = torch.distributed.all_reduce(multiplied, async_op=True)

            handle_tc.wait()
            handle_mul.wait()

            self.counter = total_count.item()

            if isinstance(self.average, torch.Tensor):
                self.average.data = multiplied / total_count
            else:
                self.average = (multiplied / total_count).item()

    def add(self, value, weight=1.0):
        """Add a value to the average"""
        self.counter += weight
        if self.average is None:
            self._init(value, weight)
        else:
            if isinstance(self.average, dict):
                for k, v in value.items():
                    self.average[k].add(v, weight)
            elif isinstance(self.average, list):
                for avg, new_value in zip(self.average, value):
                    avg.add(new_value, weight)
            else:
                self._update(value, weight)

    def _update(self, value, weight):
        alpha = float(self.update_weight * weight) / float(self.counter + self.update_weight - 1)
        if isinstance(self.average, torch.Tensor):
            self.average.mul_(1.0 - alpha)
            self.average.add_(alpha, value)
        elif isinstance(self.average, float):
            self.average *= 1.0 - alpha
            self.average += alpha * value
        else:
            raise ValueError("Unknown type")

    def _init(self, value, weight):
        if isinstance(value, dict):
            self.average = {}
            for key in value:
                self.average[key] = MeanAccumulator()
                self.average[key].add(value[key], weight)
        elif isinstance(value, list):
            self.average = []
            for v in value:
                acc = MeanAccumulator()
                acc.add(value[key], weight)
                self.average.append(acc)
        else:
            self.average = deepcopy(value)

    def reset(self):
        self.average = None
        self.counter = 0
