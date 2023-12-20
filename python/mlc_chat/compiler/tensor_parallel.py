"""Sharding operators for tensor parallelism."""
import dataclasses
from typing import Any, Dict, List

from tvm import te, tir, topi
from tvm.relax.frontend import nn


@dataclasses.dataclass
class Shard1Dim:
    """Shard a tensor by one of its dimension."""

    name: str
    shape: List[int]
    dim: int

    def gen_tir(self, shards: int, weight: nn.Tensor) -> tir.PrimFunc:
        """Generate a TIR function that shards the weight tensor by its rows."""
        assert weight.shape == self.shape
        orig_shape = [self.shape[i] if i != self.dim else self.shape[self.dim] * shards for i in range(len(self.shape)) ]
        reshape_shape = [*self.shape[: self.dim], shards, *self.shape[self.dim + 1:]]
        transpose_index = [self.dim, *range(self.dim), *range(self.dim + 1, len(self.shape))]
        w = te.placeholder(orig_shape, weight.dtype, name="w")
        reshape = topi.reshape(w, reshape_shape)
        o = topi.transpose(reshape, transpose_index)
        func = te.create_prim_func([w, o])
        return func

    def gen_shard_info(self, shards: int, weight: nn.Tensor) -> Dict[str, Any]:
        """Generate shard info for this sharding strategy."""
        assert weight.shape == self.shape
        return {
            "func": self.name,
            "out_shape": (shards, *self.shape),
            "out_dtype": weight.dtype,
        }


@dataclasses.dataclass
class RowSeg:
    """Shard a 2D tensor by its "segmented" rows, where each segment has a different number of rows
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
    rows: List[int]
    col: int
    groups: int

    @property
    def row(self) -> int:
        """Number of rows in total"""
        return sum(self.rows)

    def gen_tir(self, shards: int, weight: nn.Tensor) -> tir.PrimFunc:
        """Generate a TIR function that shards the weight tensor by its row segments."""
        assert weight.shape == [self.row, self.col]
        w = te.placeholder([self.row * shards, self.col], weight.dtype, name="w")
        ws: List[te.Tensor] = []
        offset = 0
        for idx, sub_row in enumerate(self.rows):
            assert sub_row % self.groups == 0
            ws.append(
                topi.reshape(
                    te.compute(
                        (shards * sub_row, self.col),
                        lambda i, j: w[i + offset, j],  # pylint: disable=cell-var-from-loop
                        name=f"w_{idx}",
                    ),
                    (shards, sub_row // self.groups, self.groups, self.col),
                )
            )
            offset += sub_row * shards
        o = topi.reshape(topi.concatenate(ws, axis=1), (shards, self.row, self.col))
        func = te.create_prim_func([w, o])
        return func

    def gen_shard_info(self, shards: int, weight: nn.Tensor) -> Dict[str, Any]:
        """Generate shard info for this sharding strategy."""
        assert weight.shape == [self.row, self.col]
        return {
            "func_name": self.name,
            "out_shape": (shards, self.row, self.col),
            "out_dtype": weight.dtype,
        }
