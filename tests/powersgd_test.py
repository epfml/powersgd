import torch
import torchvision

from powersgd import PowerSGD, Config


def test_no_compression_in_the_beginning():
    model = torchvision.models.resnet50()
    params = list(model.parameters())
    config = Config(
        rank=1,
        min_compression_rate=10,
        start_compressing_after_num_steps=2,
        num_iters_per_step=1,
    )
    powersgd = PowerSGD(list(params), config=config)
    for p in params:
        p.grad = torch.randn_like(p)
    grad_orig = [p.grad.clone() for p in params]
    avg_grad = powersgd.aggregate()

    for p in params:
        assert p.grad is not None
        assert p.grad.allclose(torch.zeros_like(p.grad))

    for (grad, orig) in zip(avg_grad, grad_orig):
        assert grad.allclose(orig)

    assert powersgd.step_counter == 1


def test_error_feedback_mechanism():
    torch.set_default_dtype(torch.float64)
    model = torchvision.models.resnet50()
    params = list(model.parameters())
    config = Config(
        rank=2,
        min_compression_rate=10,
        start_compressing_after_num_steps=0,
        num_iters_per_step=3,
    )
    powersgd = PowerSGD(list(params), config=config)

    for p in params:
        p.grad = torch.randn_like(p)

    grad_orig = [p.grad.clone() for p in params]
    avg_grad = powersgd.aggregate()

    for orig, avg, p in zip(grad_orig, avg_grad, params):
        assert orig.allclose(avg + p.grad)


if __name__ == "__main__":
    test_error_feedback_mechanism(model())
