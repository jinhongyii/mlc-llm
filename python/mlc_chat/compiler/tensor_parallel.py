"""Sharding operators for tensor parallelism."""
import dataclasses
from typing import Any, Dict, List

from tvm import te, tir, topi
from tvm.relax.frontend import nn


@dataclasses.dataclass
class Shard1Dim:
    """Shard a tensor by one of its dimension."""

    name: str
    dim: int

    def gen_tir(self, shards: int, weight: nn.Tensor) -> tir.PrimFunc:
        """Generate a TIR function that shards the weight tensor by its rows."""
        orig_shape = [weight.shape[i] if i != self.dim else weight.shape[self.dim] * shards for i in range(len(weight.shape)) ]
        reshape_shape = [*weight.shape[: self.dim], shards, weight.shape[self.dim], *weight.shape[self.dim + 1:]]
        transpose_index = [self.dim, *range(self.dim), *range(self.dim + 1, len(weight.shape)+1)]
        w = te.placeholder(orig_shape, weight.dtype, name="w")
        reshape = topi.reshape(w, reshape_shape)
        o = topi.transpose(reshape, transpose_index)
        func = te.create_prim_func([w, o])
        return func

    def gen_shard_info(self, shards: int, weight: nn.Tensor) -> Dict[str, Any]:
        """Generate shard info for this sharding strategy."""
        return {
            "func_name": self.name,
            "out_shape": (shards, *weight.shape),
            "out_dtype": weight.dtype,
        }


@dataclasses.dataclass
class Shard1DimSeg:
    """Shard a tensor by its "segmented" dimension, where each segment has a different shape along the dimension
    and sharded evenly on each worker.


    => Step #1:

    [#shards, rows_1 // g, g, col]
    [#shards, rows_2 // g, g, col]
    ...
    [#shards, rows_n // g, g, col]

    => Step #2:

    [#shards, sum(rows) // g, g, col]

    => Step #3:

    [#shards, sum(rows), col]

    """

    name: str
    dim: int
    segs: List[int]

    def gen_tir(self, shards: int, weight: nn.Tensor) -> tir.PrimFunc:
        """Generate a TIR function that shards the weight tensor by its row segments."""
        shape = weight.shape
        assert sum(self.segs) == shape[self.dim]
        w = te.placeholder([*shape[:self.dim], shape[self.dim] * shards, *shape[self.dim+1:]], weight.dtype, name="w")
        ws: List[te.Tensor] = []
        offset = 0
        for idx, sub_seg in enumerate(self.segs):
            ws.append(
                topi.transpose(
                    topi.reshape(
                        te.compute(
                            (*shape[:self.dim], sub_seg * shards, *shape[self.dim+1:]),
                            lambda *idx: w[idx[:self.dim]+(idx[self.dim]+offset,)+idx[self.dim+1:]],  # pylint: disable=cell-var-from-loop
                            name=f"w_{idx}",
                        ),
                        (*shape[:self.dim], shards, sub_seg, *shape[self.dim+1:]),
                    ),
                    [self.dim, *range(self.dim), *range(self.dim+1, len(shape)+1)]
                )
            )
            offset += sub_seg * shards
        o = topi.concatenate(ws, axis=1+self.dim)
        func = te.create_prim_func([w, o])
        return func

    def gen_shard_info(self, shards: int, weight: nn.Tensor) -> Dict[str, Any]:
        """Generate shard info for this sharding strategy."""
        return {
            "func_name": self.name,
            "out_shape": (shards, *weight.shape),
            "out_dtype": weight.dtype,
        }
