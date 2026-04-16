import itertools
import math
import random

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

import ntops
from ntops.kernels.scaled_dot_product_attention import CausalVariant
from tests.skippers import skip_if_cuda_not_available
from tests.utils import DEFAULT_DEVICE


def generate_arguments():
    def _generate_random_size():
        return random.randint(1, 512)

    arguments = []

    attn_mask_types = (None, torch.bool, torch.float32)
    is_causal_values = (False, True)
    scales = (None, random.uniform(0.05, 0.5))
    dtypes = (torch.float32, torch.float16)
    devices = (DEFAULT_DEVICE,)
    with_kv_cache_values = (False, True)
    causal_variants = (None, CausalVariant.LOWER_RIGHT, CausalVariant.UPPER_LEFT)

    for (
        attn_mask_type,
        is_causal,
        scale,
        dtype,
        device,
        with_kv_cache,
        causal_variant,
    ) in itertools.product(
        attn_mask_types,
        is_causal_values,
        scales,
        dtypes,
        devices,
        with_kv_cache_values,
        causal_variants,
    ):
        if attn_mask_type is not None and is_causal:
            continue

        batch_size = random.randint(1, 4)
        num_heads_q = 2 ** random.randint(1, 5)
        seq_len_q = _generate_random_size()
        head_dim = random.choice([32, 64])
        num_heads_kv = 2 ** random.randint(1, math.floor(math.log2(num_heads_q)))
        seq_len_kv = _generate_random_size()

        enable_gqa = True

        if dtype is torch.float32:
            atol = 0.01
            rtol = 0.01
        else:
            atol = 0.025
            rtol = 0.025

        if causal_variant == CausalVariant.LOWER_RIGHT and seq_len_q > seq_len_kv:
            continue

        arguments.append(
            (
                batch_size,
                num_heads_q,
                seq_len_q,
                head_dim,
                num_heads_kv,
                seq_len_kv,
                attn_mask_type,
                is_causal,
                scale,
                enable_gqa,
                causal_variant,
                with_kv_cache,
                dtype,
                device,
                rtol,
                atol,
            )
        )

    return (
        "batch_size, num_heads_q, seq_len_q, head_dim, num_heads_kv, seq_len_kv, attn_mask_type, is_causal, scale, enable_gqa, causal_variant, with_kv_cache, dtype, device, rtol, atol",
        arguments,
    )


@skip_if_cuda_not_available
@pytest.mark.parametrize(*generate_arguments())
def test_scaled_dot_product_attention(
    batch_size,
    num_heads_q,
    seq_len_q,
    head_dim,
    num_heads_kv,
    seq_len_kv,
    attn_mask_type,
    is_causal,
    scale,
    enable_gqa,
    causal_variant,
    with_kv_cache,
    dtype,
    device,
    rtol,
    atol,
):
    shape_q = (batch_size, num_heads_q, seq_len_q, head_dim)
    shape_kv = (batch_size, num_heads_kv, seq_len_kv, head_dim)

    query = torch.randn(shape_q, dtype=dtype, device=device)
    key = torch.randn(shape_kv, dtype=dtype, device=device)
    value = torch.randn(shape_kv, dtype=dtype, device=device)

    if attn_mask_type is not None:
        attn_mask = torch.rand(
            (query.shape[-2], key.shape[-2]), dtype=query.dtype, device=query.device
        )

        if attn_mask_type is torch.bool:
            attn_mask = attn_mask > 0.5
        # TODO: Non-infinite floating-point masks may cause
        # precision issues. Revisit here later.
        else:
            attn_mask = torch.where(attn_mask > 0.5, 0, float("-inf"))
            attn_mask = attn_mask.to(query.dtype)
    else:
        attn_mask = None

    key_cloned = key.clone()
    value_cloned = value.clone()

    def _generate_present_and_slot(tensor):
        present = tensor[:, :, -1:, :].clone()
        present_slot = tensor[:, :, -1:, :]
        present_slot[...] = 0

        return present, present_slot

    if with_kv_cache:
        present_key, present_key_slot = _generate_present_and_slot(key)
        present_value, present_value_slot = _generate_present_and_slot(value)
    else:
        present_key = None
        present_value = None
        present_key_slot = None
        present_value_slot = None

    ninetoothed_output = ntops.torch.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
        causal_variant=causal_variant,
        present_key=present_key,
        present_value=present_value,
        present_key_slot=present_key_slot,
        present_value_slot=present_value_slot,
    )

    if is_causal:
        if causal_variant == CausalVariant.LOWER_RIGHT:
            attn_mask = causal_lower_right(query.shape[-2], key.shape[-2])
            is_causal = False

    reference_key = key_cloned
    reference_value = value_cloned
    reference_enable_gqa = enable_gqa
    # torch_npu reference limitation:
    # float additive attn_mask with q-heads != kv-heads may fail on dim=1.
    # Work around by materializing KV head expansion and disabling GQA in reference.
    if (
        query.device.type == "npu"
        and attn_mask_type is torch.float32
        and key_cloned.shape[1] != query.shape[1]
    ):
        assert query.shape[1] % key_cloned.shape[1] == 0
        group_size = query.shape[1] // key_cloned.shape[1]
        reference_key = key_cloned.repeat_interleave(group_size, dim=1)
        reference_value = value_cloned.repeat_interleave(group_size, dim=1)
        reference_enable_gqa = False

    reference_output = F.scaled_dot_product_attention(
        query,
        reference_key,
        reference_value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=reference_enable_gqa,
    )

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
