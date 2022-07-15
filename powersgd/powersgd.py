from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, NamedTuple, Union
from contextlib2 import contextmanager

import torch

from powersgd.orthogonalization import orthogonalize
from powersgd.utils import (
    ContiguousAllocation,
    allreduce_average,
    pack,
    unpack,
    batch_unpack,
    allocate_contiguous,
)


class Aggregator(ABC):
    @abstractmethod
    def aggregate(self) -> List[torch.Tensor]:
        """
        Aggregates gradients across workers into an (approximate) average gradient.
        It reads gradient from the parameter's .grad attributes.
        This method also changes the .grad's. 
        It sets it to the compression error (for error feedback), or to zero, if there is no compression on the parameter.
        """
        pass


class AllReduce(Aggregator):
    def __init__(self, params: List[torch.Tensor]):
        self.params = params
        
    def aggregate(self) -> List[torch.Tensor]:
        assert all(p.grad is not None for p in self.params)
        gradients: list[torch.Tensor] = [p.grad for p in self.params]  # type: ignore

        if len(gradients) < 1:
            return []

        buffer, shapes = pack(gradients)
        allreduce_average(buffer)
        out = unpack(buffer, shapes)
        for g in gradients:
            g.zero_()
        return out


class Config(NamedTuple):
    rank: int  # lower rank => more aggressive compression
    min_compression_rate: float = 2  # skip compression on some gradients
    num_iters_per_step: int = 1  # lower number => more aggressive compression
    start_compressing_after_num_steps: int = 100
    use_cuda_graph: bool = True


class PowerSGD(Aggregator):
    """
    Applies PowerSGD only after a configurable number of steps,
    and only on parameters with strong compression.
    """

    def __init__(self, params: List[torch.Tensor], config: Config):
        self.config = config
        self.device = list(params)[0].device
        self.is_compressed_mask = [self._should_compress(p.shape) for p in params]

        self.step_counter = 0

        compressed_params, allreduce_params = self._split(params)
        self._powersgd = BasicPowerSGD(
            compressed_params,
            config=BasicConfig(
                rank=config.rank,
                num_iters_per_step=config.num_iters_per_step,
                use_cuda_graph=config.use_cuda_graph,
            ),
        )
        self._allreduce = AllReduce(allreduce_params)
        self._full_allreduce = AllReduce(params)

    def aggregate(self) -> List[torch.Tensor]:
        self.step_counter += 1

        if self.step_counter <= self.config.start_compressing_after_num_steps:
            return self._full_allreduce.aggregate()

        return self._merge(
            self._powersgd.aggregate(),
            self._allreduce.aggregate(),
        )

    def _split(self, params: List[torch.Tensor]):
        compressed_params = []
        uncompressed_params = []
        for param, is_compressed in zip(params, self.is_compressed_mask):
            if is_compressed:
                compressed_params.append(param)
            else:
                uncompressed_params.append(param)
        return compressed_params, uncompressed_params

    def _merge(
        self, compressed: List[torch.Tensor], uncompressed: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        assert len(compressed) + len(uncompressed) == len(self.is_compressed_mask)
        compressed_iter = iter(compressed)
        uncompressed_iter = iter(uncompressed)
        merged_list = []
        for is_compressed in self.is_compressed_mask:
            if is_compressed:
                merged_list.append(next(compressed_iter))
            else:
                merged_list.append(next(uncompressed_iter))

        return merged_list

    def _should_compress(self, shape: torch.Size) -> bool:
        return (
            shape.numel() / avg_compressed_size(shape, self.config)
            > self.config.min_compression_rate
        )


class BasicConfig(NamedTuple):
    rank: int  # lower rank => more aggressive compression
    num_iters_per_step: int = 1  # lower number => more aggressive compression
    use_cuda_graph: bool = True


class ApproximationMemory(NamedTuple):
    ps: list[torch.Tensor]
    qs: list[torch.Tensor]
    p_buffer: torch.Tensor
    q_buffer: torch.Tensor


class BasicPowerSGD(Aggregator):
    def __init__(self, params: List[torch.Tensor], config: BasicConfig):
        # Configuration
        self.config = config
        self.params = list(params)
        if len(self.params) == 0:
            return
        self.device = self.params[0].device
        self.dtype = self.params[0].dtype
        self.use_cuda_graphs = self.config.use_cuda_graph and self.device.type == "cuda"

        # State
        self.generator = torch.Generator(device=self.device).manual_seed(0)
        self.step_counter = 0

        for p in self.params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        self._inputs: list[torch.Tensor] = [p.grad for p in self.params]  # type: ignore

        self._outputs = [torch.zeros_like(param) for param in self.params]

        self._ps = allocate_contiguous(
            [self._p_shape(t) for t in self.params], self.device, self.dtype
        )
        self._qs = allocate_contiguous(
            [self._q_shape(t) for t in self.params], self.device, self.dtype
        )
        self._init_p_and_q()

        if self.use_cuda_graphs:
            self._record_cuda_graphs()

    def aggregate(self) -> List[torch.Tensor]:
        if len(self.params) == 0:
            return []

        if self.use_cuda_graphs:
            torch.cuda.synchronize()
            if self.step_counter % 2 == 0:
                self.graph_even.replay()
            else:
                self.graph_odd.replay()
        else:
            self._aggregate(
                self.step_counter,
                self._ps,
                self._qs,
                self._outputs,
            )

        self.step_counter += 1

        return self._outputs

    def _aggregate(
        self,
        step_number: int,
        ps: ContiguousAllocation,
        qs: ContiguousAllocation,
        outputs: list[torch.Tensor],
    ):
        num_iters = self.config.num_iters_per_step

        # We will do `num_iters` rounds of power iteration, 
        # and each other those iterations yields a pair of matrices P, Q
        # for each parameter in the model.
        # the P's and Q's for different parameters are stored in two contiguous buffers.
        # For each iteration (1..num_iters), 
        # we will store those buffers in `ps_found` and `qs_found`.
        # Those buffers will always contain the same data across workers.
        ps_found = self._allocate(torch.Size([num_iters, *ps.buffer.shape]))
        qs_found = self._allocate(torch.Size([num_iters, *qs.buffer.shape]))

        for it in range(num_iters):
            is_even_iteration = (step_number * num_iters + it) % 2 == 0

            for grad, p, q in zip(self._inputs, ps.tensors, qs.tensors):
                grad = view_as_matrix(grad)
                # Local matrix multiplication
                if is_even_iteration:
                    orthogonalize(q)
                    torch.matmul(grad, q, out=p)
                else:
                    orthogonalize(p)
                    torch.matmul(grad.T, p, out=q)

                # Remove the local low-rank approximation we just made
                # from the ‘error buffer’ (.grad of the parameters).
                grad.addmm_(p, q.T, alpha=-1)

            # Average the results of matrix multplication across workers.
            allreduce_average(ps.buffer if is_even_iteration else qs.buffer)

            # Store the P and Q matrices (same for all workers).
            # They will be multiplied together to get a gradient approximation later.
            ps_found[it].copy_(ps.buffer)
            qs_found[it].copy_(qs.buffer)

        # Unpack the contiguous buffers of P's and Q's found in each iteration.
        # For each, we get a Tensor containing a batch of P or Q matrices.
        pp = batch_unpack(ps_found, ps.shapes)
        qq = batch_unpack(qs_found, qs.shapes)

        # For each parameter, make a reconstruction of the gradient
        # as the outer-product of P and Q, summed over all iterations.
        # This will be the same across workers, and will be set into the 
        # self._outputs.
        for p_batch, q_batch, out in zip(pp, qq, outputs):
            out = view_as_matrix(out)
            p_batch = p_batch.permute([1, 0, 2]).reshape(p_batch.shape[1], -1)
            q_batch = q_batch.permute([1, 0, 2]).reshape(q_batch.shape[1], -1)
            torch.mm(p_batch, q_batch.T, out=out)

    def _record_cuda_graphs(self):
        with torch.cuda.device(self.device):
            with cuda_graph_warmup():
                for _ in range(3):
                    self._aggregate(
                        0,
                        self._ps,
                        self._qs,
                        self._outputs,
                    )
                    self._aggregate(
                        1,
                        self._ps,
                        self._qs,
                        self._outputs,
                    )

            # Warmup corrupts the memory. Let's reset it
            self._init_p_and_q()

            # Build a graph
            with record_cuda_graph() as self.graph_even:
                self._aggregate(
                    0,
                    self._ps,
                    self._qs,
                    self._outputs,
                )

            # We may or may not need a different graph for the odd steps,
            # if the number of iterations per SGD step is odd
            if self.config.num_iters_per_step % 2 == 1:
                with record_cuda_graph() as self.graph_odd:
                    self._aggregate(
                        1,
                        self._ps,
                        self._qs,
                        self._outputs,
                    )
            else:
                self.graph_odd = self.graph_even

    def _init_p_and_q(self):
        self._ps.buffer.copy_(
            torch.randn(
                self._ps.buffer.shape,
                generator=self.generator,
                device=self.device,
            )
        )
        self._qs.buffer.copy_(
            torch.randn(
                self._qs.buffer.shape,
                generator=self.generator,
                device=self.device,
            )
        )

    def _p_shape(self, param: torch.Tensor) -> torch.Size:
        matrix = view_as_matrix(param)
        rank = min(self.config.rank, min(matrix.shape))
        return torch.Size([matrix.shape[0], rank])

    def _q_shape(self, param: torch.Tensor) -> torch.Size:
        matrix = view_as_matrix(param)
        rank = min(self.config.rank, min(matrix.shape))
        return torch.Size([matrix.shape[1], rank])

    def _allocate(self, shape: torch.Size) -> torch.Tensor:
        return torch.empty(shape, device=self.device, dtype=self.dtype)

    @classmethod
    def _matrices_per_shape(
        cls,
        tensors: List[torch.Tensor],
    ) -> Dict[torch.Size, List[torch.Tensor]]:
        shape2tensors = defaultdict(list)
        for tensor in tensors:
            matrix = view_as_matrix(tensor)
            shape = matrix.shape
            shape2tensors[shape].append(matrix)
        return shape2tensors

    @property
    def uncompressed_num_floats(self) -> int:
        return sum(param.shape.numel() for param in self.params)

    @property
    def compressed_num_floats(self) -> float:
        return sum(avg_compressed_size(p.shape, self.config) for p in self.params)

    @property
    def compression_rate(self) -> float:
        return self.uncompressed_num_floats / self.compressed_num_floats


def view_as_matrix(tensor: torch.Tensor):
    """
    Reshape a gradient tensor into a matrix shape, where the matrix has structure
    [output features, input features].
    For a convolutional layer, this groups all "kernel" dimensions with "input features".
    """
    return tensor.view(tensor.shape[0], -1)


def avg_compressed_size(shape: torch.Size, config: Union[Config, BasicConfig]) -> float:
    rank = min(config.rank, min(shape))
    return 0.5 * config.num_iters_per_step * rank * sum(shape)

@contextmanager
def record_cuda_graph():
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        yield graph
    torch.cuda.synchronize()

@contextmanager
def cuda_graph_warmup():
    s = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(s):
        yield
