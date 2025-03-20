# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

# Required to register the custom ops
import vllm.lora.ops.xla_ops.pallas  # noqa # pylint: disable=unused-import

# N_TOKENS = [
#     8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
#     131072
# ]
# HIDDEN_SIZES = [128, 256, 512, 896, 1024, 2048, 4096, 8192, 8320]

# DTYPES = [torch.float16, torch.bfloat16]
# NUM_LORA = [1, 2, 4, 8, 16, 32]
# RANKS = [8, 16, 32, 64, 128]

N_TOKENS = [2048]
HIDDEN_SIZES = [4096]

DTYPES = [torch.bfloat16]
NUM_LORA = [1, 2, 4]
RANKS = [8]


def generate_test_data(T, D, L, N, seed, dtype=torch.float32):
    """
    Inputs: (All integers)
        T: Total number of tokens
        D: Input dim
        L: LoRA Dim
        N: N LoRAs
    
    Outputs:
        inputs:     torch.Tensor - shape (T, D)
        loras:      torch.Tensor - shape (N, 1, L, D)
        idxs:       torch.Tensor - shape (T, ) - all values must be in [0, N)
        
        ref_output: torch.Tensor - shape (T, L) - inputs @ loras[idxs].T
    """
    torch.manual_seed(seed)

    inputs = torch.randn((T, D), device="xla", dtype=dtype)
    loras = torch.randn((N, 1, L, D), device="xla", dtype=dtype)
    idxs = torch.randint(0, N, (T, ), dtype=torch.int32, device="xla")

    ref_output = ref_bgmv(inputs, loras, idxs)
    return inputs, loras, idxs, ref_output


def ref_bgmv(inputs: torch.Tensor, loras: torch.Tensor, idxs: torch.Tensor):
    selected_loras = loras[idxs]
    if len(selected_loras.shape) == 4:
        selected_loras = selected_loras.squeeze(axis=1)

    batch_size, output_size, input_size = selected_loras.shape
    return (selected_loras @ inputs.reshape(
        (batch_size, input_size, 1))).reshape((batch_size, output_size))


# Parameterize tests with various shapes and dtypes
@pytest.mark.parametrize("T", N_TOKENS)
@pytest.mark.parametrize("D", HIDDEN_SIZES)
@pytest.mark.parametrize("L", RANKS)
@pytest.mark.parametrize("N", NUM_LORA)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op_type", ["shrink", "expand"])
@pytest.mark.parametrize("seed", [0])
def test_bgmv_correctness(T, D, L, N, dtype, op_type, seed):
    if op_type == "expand":
        D, L = L, D

    inputs, loras, idxs, ref_output = generate_test_data(
        T, D, L, N, seed, dtype)

    # Run bgmv
    if op_type == "shrink":
        output = torch.ops.xla.bgmv_shrink(inputs, loras, idxs)
    else:
        output = torch.ops.xla.bgmv_expand(inputs, loras.transpose(2, 3), idxs)

    # Make sure we have no NaNs
    assert not torch.any(torch.isnan(output))

    # Compare with reference output
    assert torch.allclose(output, ref_output, rtol=1e-2, atol=1e-2)
