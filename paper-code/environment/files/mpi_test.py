import torch
import socket

torch.distributed.init_process_group("mpi")
print("{} is rank {} in world of size {}".format(socket.gethostname(), torch.distributed.get_rank(), torch.distributed.get_world_size()))

print("CPU:")
x = torch.randn([1])
torch.distributed.all_reduce(x)
print(x)

if torch.cuda.is_available():
    print("Cuda:")
    x = torch.randn([1], device='cuda')
    torch.distributed.all_reduce(x)
    print(x)
