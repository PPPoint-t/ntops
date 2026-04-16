import random

import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available
from tests.utils import generate_arguments


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_softmax(shape, dtype, device, rtol, atol):
    input = torch.randn(shape, dtype=dtype, device=device)
    dim = random.randint(0, input.ndim - 1)
    candidate_dtypes = [torch.float16, torch.float32]
    if input.device.type != "npu":
        candidate_dtypes.append(torch.float64)
    dtype = random.choice(candidate_dtypes)

    ninetoothed_output = ntops.torch.softmax(input, dim, dtype)
    reference_output = torch.nn.functional.softmax(input, dim=dim, dtype=dtype)

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
