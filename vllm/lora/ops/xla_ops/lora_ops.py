# SPDX-License-Identifier: Apache-2.0

import torch


def sgmv_expand(inputs: torch.Tensor,
                lora_b_weights: torch.Tensor,
                output_tensor: torch.Tensor,
                b_seq_start_loc: torch.Tensor,
                seq_len_tensor: torch.Tensor,
                lora_indices_tensor: torch.Tensor,
                batches: int,
                max_seq_length: int,
                token_nums: int,
                add_inputs: bool = False):
    exploded_indices = torch.repeat_interleave(lora_indices_tensor,
                                               inputs.size(0))

    bgmv_expand(inputs, lora_b_weights, output_tensor, exploded_indices,
                add_inputs)


def bgmv_expand(inputs: torch.Tensor,
                lora_b_weights: torch.Tensor,
                output_tensor: torch.Tensor,
                lora_indices_tensor: torch.Tensor,
                add_inputs: bool = True):
    selected_loras = lora_b_weights[lora_indices_tensor].to(
        dtype=output_tensor.dtype)
    if len(selected_loras.shape) == 4:
        selected_loras = selected_loras.squeeze(dim=1)
    inputs = inputs.to(dtype=output_tensor.dtype)
    # outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)
    batch_size, output_size, input_size = selected_loras.shape
    outputs = (selected_loras @ inputs.reshape(
        (batch_size, input_size, 1))).reshape((batch_size, output_size))

    limit = output_tensor.shape[0]
    if outputs.shape[0] == 1 and output_tensor.shape[0] != 1:
        limit = 1

    outputs = torch.cat(
        (outputs,
         torch.zeros((batch_size, output_tensor.shape[1] - outputs.shape[1]),
                     device=outputs.device)),
        dim=1)

    if add_inputs:
        output_tensor += outputs[:limit, :]
    else:
        output_tensor = outputs[:limit, :]


def sgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    scaling: float,
):
    exploded_indices = torch.repeat_interleave(lora_indices_tensor,
                                               inputs.size(0))

    bgmv_shrink(inputs, lora_a_weights, output_tensor, exploded_indices,
                scaling)


def bgmv_shrink(inputs: torch.Tensor,
                lora_b_weights: torch.Tensor,
                output_tensor: torch.Tensor,
                lora_indices_tensor: torch.Tensor,
                scaling: float = 1.0):
    selected_loras = lora_b_weights[lora_indices_tensor].to(
        dtype=output_tensor.dtype)
    if len(selected_loras.shape) == 4:
        selected_loras = selected_loras.squeeze(dim=1)
    inputs = inputs.to(dtype=output_tensor.dtype)
    # outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)
    batch_size, output_size, input_size = selected_loras.shape
    outputs = (selected_loras @ inputs.reshape(
        (batch_size, input_size, 1))).reshape((batch_size, output_size))

    output_tensor = scaling * outputs[:]


def sgmv_expand_slice(inputs: torch.Tensor,
                      lora_b_weights: torch.Tensor,
                      output_tensor: torch.Tensor,
                      b_seq_start_loc: torch.Tensor,
                      seq_len_tensor: torch.Tensor,
                      lora_indices_tensor: torch.Tensor,
                      batches: int,
                      max_seq_length: int,
                      token_nums: int,
                      slice_offset: int,
                      slice_size: int,
                      total_size: int,
                      add_inputs: bool = False):
    exploded_indices = torch.repeat_interleave(lora_indices_tensor,
                                               inputs.size(0))

    bgmv_expand_slice(inputs, lora_b_weights, output_tensor, exploded_indices,
                      slice_offset, slice_size, total_size, add_inputs)


def bgmv_expand_slice(inputs: torch.Tensor,
                      lora_b_weights: torch.Tensor,
                      output_tensor: torch.Tensor,
                      lora_indices_tensor: torch.Tensor,
                      slice_offset: int,
                      slice_size: int,
                      total_size: int,
                      add_inputs: bool = True):
    selected_loras = lora_b_weights[lora_indices_tensor].to(
        dtype=output_tensor.dtype)

    inputs = inputs.to(dtype=output_tensor.dtype)

    if len(selected_loras.shape) == 4:
        selected_loras = selected_loras.squeeze(dim=1)

    batch_size, output_size, input_size = selected_loras.shape

    outputs = (selected_loras @ inputs.reshape(
        (batch_size, input_size, 1))).reshape((batch_size, output_size))

    outputs = torch.cat((
        torch.zeros((batch_size, slice_offset), device=outputs.device),
        outputs,
        torch.zeros((batch_size, total_size - (slice_offset + slice_size)),
                    device=outputs.device),
    ),
                        dim=1)

    if add_inputs:
        output_tensor += outputs
    else:
        output_tensor = outputs
