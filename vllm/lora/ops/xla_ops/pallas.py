import functools
import torch
from torch.library import impl
import torch_xla
from torch_xla.experimental.custom_kernel import jax_import_guard, make_kernel_from_pallas, XLA_LIB

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu


def _bgmv_kernel(bT: int, bL: int, idx_ref, inp_ref, lora_ref, out_ref, acc_ref,
                mask_ref):

    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref[...], dtype=jnp.float32)

    t = pl.program_id(0)

    for i in range(bT):
        idx = idx_ref[i + bT * t]
        mask_ref[...] = jnp.zeros_like(mask_ref[...], dtype=jnp.float32)
        mask_ref[i, :] = jnp.ones((bL, ), dtype=jnp.float32)

        acc_ref[...] += jax.lax.dot_general(
            inp_ref[...],
            lora_ref[idx, ...], (((1, ), (1, )), ((), ())),
            preferred_element_type=jnp.float32) * mask_ref[...]

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _():
        out_ref[...] = acc_ref[...].astype(out_ref.dtype)


@jax.jit
def _bgmv(
    idxs: jax.Array,   # (T, )        int32
    inputs: jax.Array, # (T, D)       model dtype
    loras: jax.Array   # (N, 1, L, D) model dtype
) -> jax.Array:        # (T, L)       model dtype
    T, D = inputs.shape
    N, L, _ = loras.shape
    
    # TODO: Tune these
    bT = 8
    bL = 128
    bD = 128
    
    # Pad the loras' rank if it's too low. This is to allow it to fit in a TPU register
    L1 = L
    if L < bL or L % bL != 0:
        L1 = (L // bL + 1) * bL
        
    D1 = D
    if D < bD or D % bD != 0:
        D1 = (D // bD + 1) * bD
    
    T1 = T
    if T < bT or T % bT != 0:
        T1 = (T // bT + 1) * bT
    
    loras = jnp.pad(loras, ((0,0), (0,L1-L), (0,D1-D)))
    inputs = jnp.pad(inputs, ((0,T1-T), (0, D1-D)))

    return pl.pallas_call(kernel=functools.partial(_bgmv_kernel, bT, bL),
                        out_shape=jax.ShapeDtypeStruct((T1, L1),
                                                        dtype=inputs.dtype),
                        grid_spec=pltpu.PrefetchScalarGridSpec(
                            num_scalar_prefetch=1,
                            grid=(T1 // bT, L1 // bL, D1 // bD),
                            in_specs=[
                                pl.BlockSpec((bT, bD),
                                            lambda i, j, k, block_idx:
                                            (i, k)),
                                pl.BlockSpec((N, bL, bD),
                                            lambda i, j, k, block_idx:
                                            (0, j, k)),
                            ],
                            out_specs=pl.BlockSpec(
                                (bT, bL), lambda i, j, k, block_idx: (i, j)),
                            scratch_shapes=[
                                pltpu.VMEM((bT, bL), jnp.float32),
                                pltpu.VMEM((bT, bL), jnp.float32)
                            ]),
                        compiler_params=pltpu.TPUCompilerParams(
                            dimension_semantics=("parallel", "parallel", "arbitrary")),
                        name="bgmv"
            )(idxs, inputs, loras)[:T, :L]

def bgmv_shape_function(idxs, inputs, loras):
    T, _ = inputs.shape
    _, L, _ = loras.shape
    
    return [((T, L), inputs.dtype)]

XLA_LIB.define(
    "bgmv(Tensor inputs, Tensor loras, Tensor idxs) -> Tensor",
)

def ref_bgmv(inputs: jax.Array, loras: jax.Array, idxs: jax.Array):
    selected_loras = loras[idxs]    
    n_tokens, output_size, input_size = selected_loras.shape
    outputs = (
        selected_loras @ inputs.reshape((n_tokens, input_size, 1))
    ).reshape((n_tokens, output_size))
    
    return outputs

@impl(XLA_LIB, "bgmv", "XLA")
def bgmv_xla(inputs: torch.Tensor, loras: torch.Tensor, idxs: torch.IntTensor):
    inputs = inputs.to(dtype=loras.dtype)
    
    if len(loras.shape) == 4:
        loras = loras.squeeze(axis=1)
    
    _, L, D = loras.shape
    
    # FIXME: Routing the output from 1 Pallas kernel directly to another results in NaN outputs
    # so here we fallback on a reference implementation until the bug is fixed
    use_reference_on_shrink = True
    if use_reference_on_shrink and L < D:
        return ref_bgmv(inputs, loras, idxs)
    elif not use_reference_on_shrink and D < L:
        return ref_bgmv(inputs, loras, idxs)
    
    jax_import_guard()
    kernel = make_kernel_from_pallas(_bgmv, bgmv_shape_function)
    
    return kernel(idxs, inputs, loras)


@impl(XLA_LIB, "bgmv", "CompositeExplicitAutograd")
def bgmv_non_xla(inputs: torch.Tensor, loras: torch.Tensor, idxs: torch.IntTensor):
    T, _ = inputs.shape
    _, _, L, _ = loras.shape
    
    return torch.empty((T, L), device=inputs.device)
