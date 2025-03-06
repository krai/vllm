# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union

import torch

from vllm.lora.ops.xla_ops import bgmv_expand, bgmv_expand_slice, bgmv_shrink

from .punica_base import PunicaWrapperBase


# The platforms that are compatible with the PyTorch-native implementation can
# inherit this class
class PunicaWrapperTPU(PunicaWrapperBase):
    """
    PunicaWrapperTPU is designed to manage and provide metadata for the punica 
    kernel. The main function is to maintain the state information for 
    Multi-LoRA, and to provide the interface for the pytorch punica ops.
    """

    def __init__(self, max_num_batched_tokens: int, max_batches: int,
                 device: Union[torch.device, str], **kwargs):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches,
                                   device)

        # PunicaWrapperBase defines some tensors with dtype=torch.int64, which isn't supported by the TPU.
        # So convert those tensors to int32.
        # Not all of them are used by the TPU so only convert the useful ones.
        self._token_lora_indices = self._token_lora_indices.to(
            dtype=torch.int32)

    def shrink(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        scale: float,
    ):
        if self.no_lora:
            return y
        return bgmv_shrink(x, w_t_all, y, self.token_lora_indices, scale)

    def expand(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        add_inputs: bool,
    ):
        if self.no_lora:
            return y
        return bgmv_expand(x, w_t_all, y, self.token_lora_indices, add_inputs)

    def expand_slice(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        y_offset: int,
        y_slice_size: int,
        y_total_size: int,
        add_inputs: bool,
    ) -> torch.Tensor:
        if self.no_lora:
            return y
        return bgmv_expand_slice(x, w_t_all, y, self.token_lora_indices,
                                 y_offset, y_slice_size, add_inputs)

    def add_shrink(self, y: Union[Tuple[torch.Tensor, ...], torch.Tensor],
                   x: torch.Tensor, lora_a_stacked: Tuple[torch.Tensor, ...],
                   scale: float, **kwargs) -> Optional[torch.Tensor]:
        """
        Performs GEMM for multiple slices of lora_a.
            
        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale
        
        Args:
            y (Union[Tuple[torch.Tensor, ...], torch.Tensor]): Output tensors
            x (torch.Tensor): Input tensor
            lora_a_stacked (Tuple[torch.Tensor, ...]): lora_a's weights
            scale (float): Scaling factor for the operation
        """

        x = x.view(-1, x.shape[-1])

        new_y = []
        # TODO fuse these kernels
        for slice_idx in range(len(lora_a_stacked)):
            y_s = y[slice_idx]
            lora_s = lora_a_stacked[slice_idx]

            y_org = y_s
            y_s = y_s.view(-1, y_s.shape[-1])

            y_s = self.shrink(y_s, x, lora_s, scale)
            y_s = y_s.view_as(y_org)
            new_y.append(y_s)
        return tuple(new_y)

    def add_expand(self,
                   y: torch.Tensor,
                   x: Union[Tuple[torch.Tensor, ...], torch.Tensor],
                   lora_b_stacked: Tuple[torch.Tensor, ...],
                   lora_bias_stacked: Optional[Tuple[torch.Tensor, ...]],
                   output_slices: Tuple[int, ...],
                   add_inputs=True,
                   **kwargs) -> torch.Tensor:
        """
        Performs GEMM and bias addition for multiple slices of lora_b.
      
        Semantics:
            for i in range(len(lora_b_stacked)):
                slice = output_slices[i]
                y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i] + 
                    lora_bias_stacked[i] 
                offset += slice
            
        Args:
            y (torch.Tensor): Output tensor.
            x (Union[Tuple[torch.Tensor, ...], torch.Tensor]): Input tensors
            lora_b_stacked (Tuple[torch.Tensor, ...]): lora_b's weight
            lora_bias_stacked (Optional[Tuple[torch.Tensor, ...]]): 
                bias's weight
            output_slices (Tuple[int, ...]): Every slice's size
            add_inputs (bool):  Defaults to True.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        offset_left = 0
        if lora_bias_stacked is not None:
            y = self._apply_bias(self.token_lora_indices, y, output_slices,
                                 lora_bias_stacked)
        for slice_idx in range(len(lora_b_stacked)):
            y = self.expand_slice(
                y,
                x[slice_idx],
                lora_b_stacked[slice_idx],
                offset_left,
                output_slices[slice_idx],
                y_total_size=sum(output_slices),
                add_inputs=add_inputs,
            )
            offset_left += output_slices[slice_idx]
        return y.view_as(y_org)

    def add_lora_embedding(self,
                           y: torch.Tensor,
                           x: torch.Tensor,
                           lora_b_stacked: torch.Tensor,
                           add_inputs: bool = True,
                           **kwargs) -> torch.Tensor:
        """
        Applies lora  specifically for VocabParallelEmbeddingWithLoRA.

        Semantics:
            y += x @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            add_inputs (bool): Default to True.
        """

        # Embedding layer only needs the expand op
        return self.expand(y, x, lora_b_stacked, add_inputs)

    def add_lora_linear(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: Tuple[torch.Tensor, ...],
                        lora_b_stacked: Tuple[torch.Tensor, ...],
                        lora_bias_stacked: Optional[Tuple[torch.Tensor, ...]],
                        scale: float,
                        output_slices: Tuple[int, ...],
                        *,
                        buffer: Optional[Tuple[torch.Tensor, ...]] = None,
                        **kwargs) -> torch.Tensor:
        """
        Applicable to linear-related lora. 

        Semantics:
            for i in range(len(lora_a_stacked)):
                y[i] += (
                    x[i].unsqueeze(0)
                    @ lora_a_stacked[indices[i], layer_idx, :, :]
                    @ lora_b_stacked[indices[i], layer_idx, :, :]
                    * scale
                    ).squeeze(0)+lora_bias_stacked[i]

        Args:
            y (torch.Tensor): Output tensor. Will be changed in-place.
            x (torch.Tensor): Input tensor (T, E)
            lora_a_stacked (Tuple[torch.Tensor, ...]): lora_a's weight.
            lora_b_stacked (Tuple[torch.Tensor, ...]): lora_b's weight.
            lora_bias_stacked (Optional[Tuple[torch.Tensor, ...]]): lora's bias.
            scale (float): Scaling factor.
            output_slices (Tuple[int, ...]): Every slice's size.
            buffer (Optional[Tuple[torch.Tensor, ...]]): Defaults to None.
        """

        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)
        if lora_bias_stacked is not None:
            assert len(lora_bias_stacked) == len(output_slices)
            y = self._apply_bias(self.token_lora_indices, y, output_slices,
                                 lora_bias_stacked)

        if buffer is None:
            r = lora_b_stacked[0].size(-1)
            # We set the buffer to be float32 by default, consistent with the
            # triton op
            T = x.size(0)
            buffer = torch.zeros(
                (len(output_slices), T, r),
                dtype=torch.float32,
                device=x.device,
            )
        buffer = self.add_shrink(buffer, x, lora_a_stacked, scale, **kwargs)
        return self.add_expand(y,
                               buffer,
                               lora_b_stacked,
                               None,
                               output_slices,
                               add_inputs=True,
                               **kwargs)

    def add_lora_logits(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: torch.Tensor,
                        lora_b_stacked: torch.Tensor,
                        scale,
                        *,
                        buffer: Optional[torch.Tensor] = None,
                        **kwargs) -> torch.Tensor:
        """
        Applies lora  specifically for LogitsProcessorWithLoRA.
        
        Semantics:
            buffer = (x @ lora_a_stacked) * scale
            y += buffer @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_a_stacked (torch.Tensor): lora_a's weights.
            lora_b_stacked (torch.Tensor):lora_b's weights.
            scale (float): Scaling factor.
            buffer (Optional[torch.Tensor]):Default to None.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])
        r = lora_b_stacked.size(-1)
        if buffer is None:
            # We set the buffer to be float32 by default, consistent with the
            # triton op
            buffer = torch.zeros((x.size(0), r),
                                 dtype=torch.float32,
                                 device=x.device)
        # LogitsProcessorWithLoRA always using bgmv.
        buffer = bgmv_shrink(x, lora_a_stacked, buffer, self.sampler_indices,
                             scale)
        y = bgmv_expand(buffer,
                        lora_b_stacked,
                        y,
                        self.sampler_indices,
                        add_inputs=True)
        return y.view_as(y_org)

    def _update_prefill_metada(self, token_lora_tensor: torch.Tensor) -> None:
        self.batch_size = 1
        self._lora_indices_per_batch[:self.batch_size].copy_(
            token_lora_tensor[:self.batch_size])
        # TODO: .item() is extremely inefficient on TPU, so find a way around it
        self.no_lora = torch.all(token_lora_tensor == -1).item()
