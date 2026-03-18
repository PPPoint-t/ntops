import pytest
import torch


def skip_if_cuda_not_available(func):
    has_cuda = torch.cuda.is_available()
    has_npu = hasattr(torch, "npu") and torch.npu.is_available()

    return pytest.mark.skipif(
        not has_cuda and not has_npu, 
        reason="Neither CUDA nor NPU is available on this machine"
    )(func)
