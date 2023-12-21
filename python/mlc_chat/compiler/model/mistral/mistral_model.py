"""
Implementation for Mistral architecture.
"""
import dataclasses
import math
from typing import Any, Dict, Optional

import tvm
from tvm import relax as rx
from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.script import tir as T
from tvm.topi.cuda.scan import exclusive_scan, inclusive_scan
from tvm.topi.cuda.sort import topk
from ....support import logging
from ....support.config import ConfigBase
from ....support.style import bold
from ... import tensor_parallel as tp

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MistralConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Mistral model."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_experts: int
    num_experts_per_token: int
    num_hidden_layers: int
    rms_norm_eps: float
    vocab_size: int
    position_embedding_base: int = 0
    context_window_size: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    sliding_window_size: int = 4096
    prefill_chunk_size: int = 0
    attention_sink_size: int = 4
    tensor_parallel_shards: int = 1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.context_window_size == 0:
            for name in ["max_position_embeddings", "max_sequence_length"]:
                if name in self.kwargs:
                    self.context_window_size = self.kwargs.pop(name)
                    logger.info(
                        "%s not found in config.json. Falling back to %s (%d)",
                        bold("context_window_size"),
                        bold(name),
                        self.context_window_size,
                    )
                    break
            else:
                raise ValueError(
                    "Unable to determine the maxmimum sequence length, because none of "
                    "`context_window_size`, `max_position_embeddings` or `max_sequence_length` is "
                    "provided in `config.json`."
                )
        if self.position_embedding_base == 0:
            if "rope_theta" in self.kwargs:
                self.position_embedding_base = self.kwargs.pop("rope_theta")
            else:
                self.position_embedding_base = 10000
        if self.num_key_value_heads == 0:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.head_dim * self.num_attention_heads == self.hidden_size

        assert self.attention_sink_size >= 0

        if self.prefill_chunk_size == 0:
            # chunk size same as sliding window by default
            self.prefill_chunk_size = self.sliding_window_size
        self.context_window_size = -1
        logger.info(
            "Using sliding window attention, setting %s to -1",
            bold("context_window_size"),
        )


# pylint: disable=invalid-name,missing-docstring


class RotaryEmbedding(nn.Module):
    """Cache relative Rotary Embedding."""

    def __init__(self, config: MistralConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.position_embedding_base = config.position_embedding_base

    def forward(self, q: Tensor, k: Tensor, q_offset: tir.Var):
        def te_op(x: te.Tensor, offset: tir.Var):
            dtype = x.dtype

            def compute(b: tir.Var, s: tir.Var, h: tir.Var, d: tir.Var):
                head_dim = tir.const(self.head_dim, "int32")
                position_embedding_base = tir.const(self.position_embedding_base, "float32")
                freq = tir.power(
                    position_embedding_base,
                    (d * 2 % head_dim).astype("float32") / head_dim,
                )
                freq = (offset + s) / freq
                cos = tir.cos(freq).astype(dtype) * x[b, s, h, d]
                sin = tir.sin(freq).astype(dtype) * tir.if_then_else(
                    d < head_dim // 2,
                    -x[b, s, h, d + head_dim // 2],
                    x[b, s, h, d - head_dim // 2],
                )
                return cos + sin

            return te.compute(x.shape, compute, name="rotary")

        q_embed = op.tensor_expr_op(te_op, "rotary_embedding", args=[q, q_offset])
        k_embed = op.tensor_expr_op(te_op, "rotary_embedding", args=[k, 0])
        return q_embed, k_embed


class MistralMLP(nn.Module):
    """Same as in Llama architecture (LlamaFFN)."""

    def __init__(self, config: MistralConfig):
        super().__init__()
        self.intermediate_size = config.intermediate_size // config.tensor_parallel_shards
        self.gate_up_proj = nn.Linear(
            in_features=config.hidden_size,
            out_features=2 * self.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor):
        concat_x1_x2 = self.gate_up_proj(x)
        x1, x2 = op.split(concat_x1_x2, 2, axis=-1)
        return self.down_proj(op.silu(x1) * x2)


class MistralExperts(nn.Module):
    def __init__(self, num_experts, num_experts_per_token, in_features, out_features):
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token  
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter((num_experts, out_features, in_features))
        self.dtype = "float32"
        self.cnt = 0

    def gemv_e1_e3(self, x: Tensor, indptr: Tensor, ):

        @T.prim_func
        def _gemv_e1_e3(var_x: T.handle, var_w: T.handle, var_indptr: T.handle, var_o: T.handle):
            T.func_attr({"op_pattern": 4})
            x = T.match_buffer(var_x, (1, self.in_features), self.dtype)
            w = T.match_buffer(var_w, (self.num_experts, self.out_features, self.in_features), self.dtype)
            indptr = T.match_buffer(var_indptr, (self.num_experts_per_token,), "int32")
            o = T.match_buffer(var_o, (self.num_experts_per_token, self.out_features), self.dtype)
            # with T.block("root"):
            for expert_id in T.thread_binding(self.num_experts_per_token, thread="blockIdx.y"):
                with T.block("gemv_o"):
                    v_expert_id_o = T.axis.spatial(self.num_experts_per_token, expert_id)
                    vi_o = T.axis.spatial(1, 0)
                    vj_o = T.axis.reduce(1, 0)
                    T.reads(x[0, 0:self.in_features], w[indptr[v_expert_id_o], 0:self.out_features, 0:self.in_features], indptr[v_expert_id_o])
                    T.writes(o[v_expert_id_o, 0:self.out_features])
                    for i, j in T.grid(self.out_features, self.in_features):
                        with T.block("gemv"):
                            vi_i, vj_i = T.axis.remap("SR", [i, j])
                            T.reads(x[0, vj_i], w[indptr[v_expert_id_o], vi_i, vj_i], indptr[v_expert_id_o])
                            T.writes(o[v_expert_id_o, vi_i])
                            with T.init():
                                o[v_expert_id_o, vi_i] = T.cast(T.float16(0), self.dtype)
                            o[v_expert_id_o, vi_i] = o[v_expert_id_o, vi_i] + x[0, vj_i] * w[indptr[v_expert_id_o], vi_i, vj_i]
                            
        bb = rx.BlockBuilder.current()
        gvar = bb.add_func(_gemv_e1_e3, "gemv_e1_e3")
        return op.wrap_nested(
            bb.emit(
                rx.call_tir(
                    gvar,
                    [x._expr, self.weight._expr, indptr._expr],
                    out_sinfo=rx.TensorStructInfo(
                        [indptr.shape[0], self.out_features], self.dtype
                    ),
                )
            ),
            name="gemv_e1_e3",
        )

    def gemv_e2(self, x: Tensor, indptr: Tensor):

        @T.prim_func
        def _gemv_e2(var_x: T.handle, var_w: T.handle, var_indptr: T.handle, var_o: T.handle):
            T.func_attr({"op_pattern": 4})
            x = T.match_buffer(var_x, (self.num_experts_per_token, self.in_features), self.dtype)
            w = T.match_buffer(var_w, (self.num_experts, self.out_features, self.in_features), self.dtype)
            indptr = T.match_buffer(var_indptr, (self.num_experts_per_token,), "int32")
            o = T.match_buffer(var_o, (self.num_experts_per_token, self.out_features), self.dtype)
            # with T.block("root"):
            for expert_id in T.thread_binding(self.num_experts_per_token, thread="blockIdx.y"):
                with T.block("gemv_o"):
                    v_expert_id_o = T.axis.spatial(self.num_experts_per_token, expert_id)
                    vi_o = T.axis.spatial(1, 0)
                    vj_o = T.axis.reduce(1, 0)
                    T.reads(x[v_expert_id_o, 0:self.in_features], w[indptr[v_expert_id_o], 0:self.out_features, 0:self.in_features], indptr[v_expert_id_o])
                    T.writes(o[v_expert_id_o, 0:self.out_features])
                    for i, j in T.grid(self.out_features, self.in_features):
                        with T.block("gemv"):
                            vi_i, vj_i = T.axis.remap("SR", [i, j])
                            T.reads(x[v_expert_id_o, vj_i], w[indptr[v_expert_id_o], vi_i, vj_i], indptr[v_expert_id_o])
                            T.writes(o[v_expert_id_o, vi_i])
                            with T.init():
                                o[v_expert_id_o, vi_i] = T.cast(T.float16(0), self.dtype)
                            o[v_expert_id_o, vi_i] = o[v_expert_id_o, vi_i] + x[v_expert_id_o, vj_i] * w[indptr[v_expert_id_o], vi_i, vj_i]
                            
        bb = rx.BlockBuilder.current()
        gvar = bb.add_func(_gemv_e2, "gemv_e2")
        return op.wrap_nested(
            bb.emit(
                rx.call_tir(
                    gvar,
                    [x._expr, self.weight._expr, indptr._expr],
                    out_sinfo=rx.TensorStructInfo(
                        [indptr.shape[0], self.out_features], self.dtype
                    ),
                )
            ),
            name="gemv_e2",
        )
        
    def group_gemm(self, input: nn.Tensor, weight: nn.Tensor, indptr: nn.Tensor):

        Ne = self.num_experts
        N = self.out_features
        K = self.in_features

        BLK_M = 8
        BLK_N = 128
        BLK_K = 32

        TX = 8
        TY = 32
        CTA_COUNT = 1024

        VEC_X = 1
        VEC_W = 1
        VEC_O = 1
        VEC_DOT = 1

        UNROLL = 64
        STORAGE_ALIGN = False

        assert BLK_K % 8 == 0

        # fmt: off
        @T.prim_func(private=True)
        def group_gemm(
            var_X: T.handle,
            weight: T.Buffer((Ne, N, K), dtype=self.dtype),
            indptr: T.Buffer((Ne + 1), dtype="int32"),
            var_O: T.handle,
        ):
            T.func_attr({"tir.is_scheduled": 1})
            B = T.int32(is_size_var=True)
            X = T.match_buffer(var_X, (B, K), self.dtype)
            O = T.match_buffer(var_O, (B, N), self.dtype)

            for i in T.thread_binding(CTA_COUNT, thread="blockIdx.x"):
                with T.block("CTA"):
                    bx = T.axis.spatial(CTA_COUNT, i)

                    sum = T.alloc_buffer((2,), "int32", scope="local")
                    row = T.alloc_buffer((2,), "int32", scope="local")
                    cur_e = T.alloc_buffer((1,), "int32", scope="local")
                    cur_tile_cnt = T.alloc_buffer((1,), "int32", scope="local")

                    tile_per_row = T.ceildiv(N, BLK_N)
                    sum[0] = 0
                    sum[1] = T.ceildiv(indptr[1] - indptr[0], BLK_M) * tile_per_row
                    row[0] = 0
                    row[1] = indptr[1] - indptr[0]
                    cur_e[0] = 0
                    cur_tile_cnt[0] = bx
                    row[0] = 0
            
                    while cur_e[0] < Ne:
                        # move to the current group
                        while cur_tile_cnt[0] >= sum[1] and cur_e[0] < Ne:
                            cur_e[0] += 1
                            if cur_e[0] < Ne:
                                a: T.int32 = cur_e[0]
                                delta: T.int32 = indptr[a + 1] - indptr[a]
                                sum[0] = sum[1]
                                sum[1] += T.ceildiv(delta, BLK_M) * tile_per_row
                                row[0] = row[1]
                                row[1] += delta
                        
                        # sync threads to make sure all threads have the same tile position
                        T.evaluate(T.Call(None, "tir.tvm_storage_sync", tvm.runtime.convert(["shared"])))
                        
                        if (cur_e[0] < Ne):
                            # fetch current tile position
                            a: T.int32 = cur_e[0]
                            delta: T.int32 = indptr[a + 1] - indptr[a]
                            tile_cnt_in_group: T.int32 = cur_tile_cnt[0] - sum[0]
                            tile_m: T.int32 = T.floordiv(tile_cnt_in_group, tile_per_row)
                            tile_n: T.int32 = T.floormod(tile_cnt_in_group, tile_per_row)
                            
                            tile_m_start: T.int32 = row[0] + tile_m * BLK_M
                            tile_n_start: T.int32 = tile_n * BLK_N

                            with T.block("gemm"):
                                X_tile = T.alloc_buffer((BLK_M, K), self.dtype, scope="shared")
                                W_tile = T.alloc_buffer((BLK_N, K), self.dtype, scope="shared")
                                O_tile = T.alloc_buffer((BLK_M, BLK_N), "float32", scope="local")
                                
                                for a0, a1 in T.grid(BLK_M, K): 
                                    with T.block("X_shared"):
                                        i, j = T.axis.remap("SS", [a0, a1])
                                        X_tile[i, j] = T.if_then_else(tile_m_start + i < row[1], X[tile_m_start + i, j], tir.const(0, self.dtype))
                                for a0, a1 in T.grid(BLK_N, K):
                                    with T.block("W_shared"):
                                        i, j = T.axis.remap("SS", [a0, a1])
                                        n: T.int32 = tile_n_start + i
                                        W_tile[i, j] = T.if_then_else(
                                            n < N, 
                                            weight[a, n, j],
                                            tir.const(0, self.dtype)
                                        )
                                for a0, a1, a2 in T.grid(BLK_M, BLK_N, K):
                                    with T.block("compute"):
                                        i, j, k = T.axis.remap("SSR", [a0, a1, a2])
                                        with T.init():
                                            O_tile[i, j] = tir.const(0, "float32")
                                        O_tile[i, j] += T.cast(X_tile[i, k], "float32") *  T.cast(W_tile[j, k], "float32")
                                for a0, a1 in T.grid(BLK_M, BLK_N):
                                    with T.block("store"):
                                        i, j = T.axis.remap("SS", [a0, a1])
                                        if tile_m_start + i < row[1] and tile_n_start + j < N:
                                            O[tile_m_start + i, tile_n_start + j] = O_tile[i, j]
                        # move to next tile
                        cur_tile_cnt[0] += CTA_COUNT
        # fmt: on

        sch = tvm.tir.Schedule(group_gemm)

        main_block = sch.get_block("compute")
        x, y, k = sch.get_loops(main_block)

        ty, yi = sch.split(y, [TY, None])
        tx, xi, vec_c = sch.split(x, [TX, None, VEC_DOT])
        ko, ki = sch.split(k, factors=[None, BLK_K])
        sch.reorder(ty, tx, ko, ki, yi, xi, vec_c)
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(vec_c)

        if UNROLL > 0:
            sch.annotate(tx, ann_key="pragma_auto_unroll_max_step", ann_val=UNROLL)
            sch.annotate(tx, ann_key="pragma_unroll_explicit", ann_val=1)

        l2g = sch.get_block("store")
        sch.reverse_compute_at(l2g, tx, preserve_unit_loops=True)
        _, v = sch.split(sch.get_loops(l2g)[-1], [None, VEC_O])
        sch.vectorize(v)


        def _cooperative_fetch(block, vec_len):
            num_loops = len(sch.get_loops(block))
            sch.compute_at(block, ko, preserve_unit_loops=True)
            loops = sch.get_loops(block)[-num_loops:]
            ty, tx, _, vec = sch.split(
                sch.fuse(*loops),
                factors=[TY, TX, None, vec_len],
            )
            sch.vectorize(vec)
            sch.bind(ty, "threadIdx.y")
            sch.bind(tx, "threadIdx.x")
            if STORAGE_ALIGN:
                sch.storage_align(block, 0, axis=1, factor=8, offset=vec_len)
            return block


        a_g2s = _cooperative_fetch(sch.get_block("X_shared"), vec_len=VEC_X)
        b_g2s = _cooperative_fetch(sch.get_block("W_shared"), vec_len=VEC_W)

        sch.decompose_reduction(main_block, ko)

        func = sch.mod["main"]
        bb = rx.BlockBuilder.current()
        self.cnt +=1
        gvar = bb.add_func(func, "group_gemm_"+str(self.cnt))
        return nn.op.wrap_nested(
            bb.emit(
                rx.call_tir(
                    gvar,
                    [input._expr, weight._expr, indptr._expr],
                    out_sinfo=rx.TensorStructInfo(
                        [input.shape[0], self.out_features], self.dtype
                    ),
                )
            ),
            name="group_gemm_"+str(self.cnt),
        )
        
        
    def forward(self, x: Tensor, indptr: Tensor, single_batch_decode: bool = False):
        assert x.ndim == 2
        if single_batch_decode:
            #single-batch decode
            assert x.shape[1] == self.in_features
            assert indptr.ndim == 1
            if x.shape[0] == 1:
                return self.gemv_e1_e3(x, indptr)
            else:
                return self.gemv_e2(x, indptr)
        
        return self.group_gemm(x, self.weight, indptr)


class MistralMoE(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.gate = nn.Linear(
            in_features=config.hidden_size, out_features=config.num_experts, bias=False
        )
        self.num_experts_per_token = config.num_experts_per_token
        self.num_experts = config.num_experts
        self.intermediate_size = config.intermediate_size // config.tensor_parallel_shards
        self.gate_up_proj = MistralExperts(
            self.num_experts,
            self.num_experts_per_token,
            in_features=config.hidden_size,
            out_features=2 * self.intermediate_size,
        )
        self.down_proj = MistralExperts(
            self.num_experts,
            self.num_experts_per_token,
            in_features=self.intermediate_size,
            out_features=config.hidden_size,
        )
        self.dtype = "float32"

    def topk(self, x, k):
        index_dtype = "int32"

        @T.prim_func
        def top2_func(
            x_handle: T.handle,
            out_handle: T.handle,
            out_index_handle: T.handle,
        ) -> None:
            total_rows = T.int64()
            x = T.match_buffer(x_handle, (total_rows, self.num_experts), self.dtype)
            out = T.match_buffer(out_handle, (total_rows, 2), self.dtype)
            out_index = T.match_buffer(out_index_handle, (total_rows, 2), index_dtype)
            local_top_k = T.alloc_buffer((2,), dtype=self.dtype, scope="local")
            local_top_k_index = T.alloc_buffer((2,), dtype=index_dtype, scope="local")
            T.func_attr({"tir.noalias": True, "tir.is_scheduled": True})
            for io in T.thread_binding(0, T.ceildiv(total_rows, T.int64(1024)), "blockIdx.x"):
                for ii in T.thread_binding(0, T.min(total_rows, T.int64(1024)), "threadIdx.x"):
                    with T.block("top2"):
                        vi = T.axis.spatial(total_rows, io * T.int64(1024) + ii)
                        T.where(io * T.int64(1024) + ii < total_rows)
                        with T.block("init"):
                            local_top_k[0] = T.min_value(self.dtype)
                            local_top_k_index[0] = 0
                        for k in range(self.num_experts):
                            with T.block("update"):
                                vk = T.axis.remap("S", [k])
                                if x[vi, vk] > local_top_k[0]:
                                    local_top_k[1] = local_top_k[0]
                                    local_top_k_index[1] = local_top_k_index[0]
                                    local_top_k[0] = x[vi, vk]
                                    local_top_k_index[0] = vk
                                elif x[vi, vk] > local_top_k[1]:
                                    local_top_k[1] = x[vi, vk]
                                    local_top_k_index[1] = vk
                        for j in T.unroll(2):
                            with T.block("output"):
                                vj = T.axis.remap("S", [j])
                                out[vi, vj] = local_top_k[vj]
                                out_index[vi, vj] = local_top_k_index[vj]

        if k != 2:
            return op.tensor_expr_op(topk, "topk", args=[x, k, -1, "both", False, "int32"])
        bb = rx.BlockBuilder.current()
        gvar = bb.add_func(top2_func, "top2")
        return op.wrap_nested(
            bb.emit(
                rx.call_tir(
                    gvar,
                    [x._expr],
                    out_sinfo=[
                    rx.TensorStructInfo([x.shape[0], k], self.dtype),
                    rx.TensorStructInfo([x.shape[0], k], index_dtype),
                ],
                )
            ),
            name="flattened_expert_indices",
        )


    def cumsum(self, data: Tensor, dim: int) -> Tensor:
        return op.tensor_expr_op(inclusive_scan, "cumsum", args=[data, dim, "int32"])

    def topk_mask(self, topk_indices: Tensor) -> Tensor:
        from functools import reduce

        def te_topk_mask_op(topk_indices):
            ntokens = topk_indices.shape[0]
            assert topk_indices.shape[1] == self.num_experts_per_token
            return te.compute(
                (ntokens, self.num_experts),
                lambda i, j: tir.expr.Select(
                    reduce(
                        lambda a, b: tir.Or(a, b),
                        [topk_indices[i, k] == j for k in range(self.num_experts_per_token)],
                    ),
                    true_value=tir.const(1, "int32"),
                    false_value=tir.const(0, "int32"),
                ),
            )

        return op.tensor_expr_op(te_topk_mask_op, "topk_mask", args=[topk_indices])

    def get_indices(self, cumsum_colwise_flattened: Tensor, expert_indices: Tensor) -> Tensor:

        @T.prim_func
        def get_flattened_expert_indices_scheduled(
            var_cumsum_colwise_flattened: T.handle,
            var_expert_indices: T.handle,
            var_flattened_expert_indices: T.handle,
        ):
            T.func_attr({"tir.is_scheduled": 1})
            batch_size = T.SizeVar("batch_size", "int32")
            cumsum_flattened_length = T.SizeVar("cumsum_flattened_length", "int32")

            cumsum_colwise_flattened = T.match_buffer(
                var_cumsum_colwise_flattened, shape=[cumsum_flattened_length], dtype="int32"
            )
            expert_indices = T.match_buffer(
                var_expert_indices, shape=[batch_size, self.num_experts_per_token], dtype="int32"
            )
            flattened_expert_indices = T.match_buffer(
                var_flattened_expert_indices,
                shape=[batch_size * self.num_experts_per_token],
                dtype="int32",
            )

            for io in T.thread_binding(
                0, T.floordiv(cumsum_flattened_length, T.int32(1024)), "blockIdx.x"
            ):
                for ii in T.thread_binding(0, T.int32(1024), "threadIdx.x"):
                    with T.block("get_indices"):
                        vi = T.axis.spatial(cumsum_flattened_length, io * T.int32(1024) + ii)
                        T.where(io * T.int32(1024) + ii < cumsum_flattened_length)
                        T.reads(
                            cumsum_colwise_flattened[vi - 1 : vi - 1 + 2], expert_indices[:, 0:2]
                        )
                        T.writes(flattened_expert_indices[:])
                        expert_idx = T.alloc_buffer(shape=(), dtype="int32", scope="local")
                        if (
                            vi == 0 and cumsum_colwise_flattened[vi] > 0
                        ) or cumsum_colwise_flattened[vi] != cumsum_colwise_flattened[vi - 1]:
                            idx: T.SizeVar("idx", "int32") = cumsum_colwise_flattened[vi] - 1
                            instance_id: T.SizeVar("instance_id", "int32") = T.truncmod(
                                vi, batch_size
                            )
                            expert_id: T.SizeVar("expert_id", "int32") = T.truncdiv(vi, batch_size)
                            for j in T.serial(0, self.num_experts_per_token):
                                with T.block("select_expert"):
                                    vj = T.axis.spatial(self.num_experts_per_token, j)
                                    vinstance_id = T.axis.spatial(batch_size, instance_id)
                                    vexpert_id = T.axis.spatial(
                                        T.truncdiv(cumsum_flattened_length, batch_size), expert_id
                                    )
                                    if expert_indices[vinstance_id, vj] == vexpert_id:
                                        expert_idx[()] = vj
                            flattened_expert_indices[idx] = (
                                instance_id * self.num_experts_per_token + expert_idx[()]
                            )

        bb = rx.BlockBuilder.current()
        gvar = bb.add_func(get_flattened_expert_indices_scheduled, "get_flattened_expert_indices")
        return op.wrap_nested(
            bb.emit(
                rx.call_tir(
                    gvar,
                    [cumsum_colwise_flattened._expr, expert_indices._expr],
                    out_sinfo=rx.TensorStructInfo(
                        [expert_indices.shape[0] * self.num_experts_per_token], "int32"
                    ),
                )
            ),
            name="flattened_expert_indices",
        )

    def get_indptr(self, cumsum_colwise_flattened: Tensor) -> Tensor:

        @T.prim_func
        def get_expert_instance_indptr(
            var_cumsum_colwise_flattened: T.handle,
            var_expert_instance_indptr: T.handle,
            batch_size: T.int32,
        ):
            cumsum_colwise_flattened = T.match_buffer(
                var_cumsum_colwise_flattened, shape=[batch_size * self.num_experts], dtype="int32"
            )
            expert_instance_indptr = T.match_buffer(
                var_expert_instance_indptr, shape=[self.num_experts + 1], dtype="int32"
            )

            for expert_id in T.serial(0, self.num_experts + 1):
                with T.block("indptr"):
                    vexpert_id = T.axis.spatial(self.num_experts + 1, expert_id)
                    expert_instance_indptr[vexpert_id] = T.Select(
                        condition=vexpert_id > 0,
                        true_value=cumsum_colwise_flattened[vexpert_id * batch_size - 1],
                        false_value=T.int32(0),
                    )

        bb = rx.BlockBuilder.current()
        gvar = bb.add_func(get_expert_instance_indptr, "get_expert_instance_indptr")
        return op.wrap_nested(
            bb.emit(
                rx.call_tir(
                    gvar,
                    [cumsum_colwise_flattened._expr],
                    out_sinfo=rx.TensorStructInfo([self.num_experts + 1], "int32"),
                    tir_vars=[cumsum_colwise_flattened.shape[0] // self.num_experts],
                )
            ),
            name="expert_instance_indptr",
        )

    def scatter_output(self, flattened_indices: Tensor, linear_out: Tensor) -> Tensor:

        @T.prim_func
        def tir_scatter_output(
            var_unscattered_output: T.handle,
            var_flattened_expert_indices: T.handle,
            var_scattered_output: T.handle,
        ):
            out_features = T.int64()
            flattened_indices_length = T.int64()

            unscattered_output = T.match_buffer(
                var_unscattered_output,
                shape=[flattened_indices_length, out_features],
                dtype=self.dtype,
            )
            flattened_expert_indices = T.match_buffer(
                var_flattened_expert_indices, shape=[flattened_indices_length], dtype="int32"
            )
            scattered_output = T.match_buffer(
                var_scattered_output,
                shape=[flattened_indices_length, out_features],
                dtype=self.dtype,
            )

            for i in T.serial(0, flattened_indices_length):
                for j in T.serial(0, out_features):
                    with T.block("scatter"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        scattered_output[flattened_expert_indices[vi], vj] = unscattered_output[
                            vi, vj
                        ]

        bb = rx.BlockBuilder.current()
        gvar = bb.add_func(tir_scatter_output, "scatter_output")
        return op.wrap_nested(
            bb.emit(
                rx.call_tir(
                    gvar,
                    [linear_out._expr, flattened_indices._expr],
                    out_sinfo=linear_out._expr.struct_info,
                )
            ),
            name="scatter_output",
        )

    def sum(self, x):
        if self.num_experts_per_token == 2:
            def te_add(x):
                new_shape = (x.shape[0], x.shape[2])
                return te.compute(
                    new_shape,
                    lambda i, j: x[i, 0, j] + x[i, 1, j],
                    name="add",
                )
            return op.tensor_expr_op(te_add, "topk_mask", args=[x])
        else:
            return op.sum(x, axis=1)
    def forward(self, x: Tensor):

        assert x.ndim == 3
        input_shape = x.shape
        x = op.reshape(x, (input_shape[0] * input_shape[1], input_shape[2]))
        num_tokens = input_shape[0] * input_shape[1]

        # MoE data preparation
        gate: Tensor = self.gate(x)
        expert_weights, expert_indices = self.topk(gate, self.num_experts_per_token)
        expert_weights = op.softmax(expert_weights, axis=-1)
        if num_tokens == 1:
            #single batch decode
            expert_indices = op.reshape(expert_indices, (self.num_experts_per_token,))
            concat_x1_x3 = self.gate_up_proj(x, expert_indices, single_batch_decode=True)
            x1, x3 = op.split(concat_x1_x3, indices_or_sections=2, axis=-1)
            linear_out = self.down_proj(op.silu(x1) * x3, expert_indices, single_batch_decode=True)
            unflattened = op.reshape(linear_out, (num_tokens, self.num_experts_per_token, linear_out.shape[-1]))
        else:
            expert_mask = self.topk_mask(expert_indices)
            mask_T_flattened = op.reshape(
                op.permute_dims(expert_mask), (expert_mask.shape[0] * expert_mask.shape[1],)
            )
            cumsum_colwise_flattened = self.cumsum(mask_T_flattened, dim=0)
            flattened_indices = self.get_indices(cumsum_colwise_flattened, expert_indices)
            indptr = self.get_indptr(cumsum_colwise_flattened)
            token_indices = op.divide(flattened_indices, Tensor.from_const(self.num_experts_per_token))
            gathered_x = op.take(x, token_indices, axis=0)

            # MLP forward begin
            concat_x1_x3 = self.gate_up_proj(gathered_x, indptr)
            x1, x3 = op.split(concat_x1_x3, indices_or_sections=2, axis=-1)
            linear_out = self.down_proj(op.silu(x1) * x3, indptr)
            # MLP forward end

            # MoE result post-processing
            unpermuted = self.scatter_output(flattened_indices, linear_out)
            unflattened = op.reshape(
                unpermuted, (num_tokens, self.num_experts_per_token, unpermuted.shape[1])
            )
        expert_weights = op.reshape(expert_weights, (num_tokens, self.num_experts_per_token, 1))
        weighted_sum = self.sum(unflattened * expert_weights)
        weighted_sum = op.reshape(
            weighted_sum, (input_shape[0], input_shape[1], weighted_sum.shape[-1])
        )
        return weighted_sum


class MistralAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Same as LlamaAttention, but with sliding window attention using a rolling buffer cache."""

    def __init__(self, config: MistralConfig, rotary_embedding: RotaryEmbedding):
        self.rotary_embedding = rotary_embedding
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_q_heads = config.num_attention_heads // config.tensor_parallel_shards
        self.num_kv_heads = config.num_key_value_heads // config.tensor_parallel_shards
        self.sliding_window_size = config.sliding_window_size
        self.attention_sink_size = config.attention_sink_size
        self.qkv_proj = nn.Linear(
            in_features=config.hidden_size,
            out_features=(self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_q_heads * self.head_dim, config.hidden_size, bias=False)
        self.k_cache = RollingKVCacheWithSinks(
            self.sliding_window_size, [self.num_kv_heads, self.head_dim]
        )
        self.v_cache = RollingKVCacheWithSinks(
            self.sliding_window_size, [self.num_kv_heads, self.head_dim]
        )

    def interleave_kv(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        k_cur: Tensor,
        v_cur: Tensor,
        kv_seq_len: tir.Var,
        rolling_cache_len: tir.Var,
        cache_offset: tir.Var,
    ):
        """Unrotate and concatenate currunt and cached k and v"""
        h_kv, d = self.num_kv_heads, self.head_dim
        kv_s, c, o = kv_seq_len, rolling_cache_len, cache_offset
        b = k_cur.shape[0]

        k_cached = op.reshape(self.k_cache.view(c), (b, c, h_kv, d))
        v_cached = op.reshape(self.v_cache.view(c), (b, c, h_kv, d))

        def _cache_unrotate(x_cached, rolling_cache_len, cache_offset):
            return te.compute(
                (b, kv_s, h_kv, d),
                lambda xb, xs, xh, xd: te.if_then_else(
                    xs < self.attention_sink_size,
                    x_cached[xb, xs, xh, xd],
                    te.if_then_else(
                        xs < rolling_cache_len - cache_offset + self.attention_sink_size,
                        x_cached[xb, xs + cache_offset - self.attention_sink_size, xh, xd],
                        x_cached[xb, xs + cache_offset - rolling_cache_len, xh, xd],
                    ),
                ),
                name="cache_unrotate_te",
            )

        def _cache_cur_concat(x_cached, x_cur, rolling_cache_len):
            return te.compute(
                (b, kv_s, h_kv, d),
                lambda xb, xs, xh, xd: te.if_then_else(
                    xs < rolling_cache_len,
                    x_cached[xb, xs, xh, xd],
                    x_cur[xb, xs - rolling_cache_len, xh, xd],
                ),
                name="cache_cur_concat_te",
            )

        k_cached = op.tensor_expr_op(
            _cache_unrotate,
            name_hint="te_cache_unrotate_key",
            args=[k_cached, c, o],
        )
        k = op.tensor_expr_op(
            _cache_cur_concat,
            name_hint="te_cache_cur_concat_key",
            args=[k_cached, k_cur, c],
        )

        v_cached = op.tensor_expr_op(
            _cache_unrotate,
            name_hint="te_cache_unrotate_value",
            args=[v_cached, c, o],
        )
        v = op.tensor_expr_op(
            _cache_cur_concat,
            name_hint="te_cache_cur_concat_value",
            args=[v_cached, v_cur, c],
        )

        self.k_cache.override(
            op.squeeze(k_cur, axis=0), self.sliding_window_size, self.attention_sink_size
        )
        self.v_cache.override(
            op.squeeze(v_cur, axis=0), self.sliding_window_size, self.attention_sink_size
        )

        return k, v

    def forward(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        rolling_cache_len: tir.Var,  # Number of elements currently in the cache.
        kv_seq_len: tir.Var,  # Equals to ``seq_len + rolling_cache_len``.
        cache_offset: tir.Var,
    ):
        """Forward pass of MistralAttention, performing QKV."""
        d, h_q, h_kv = self.head_dim, self.num_q_heads, self.num_kv_heads
        b, s, _ = hidden_states.shape
        assert b == 1, "Only support batch size 1 at this moment."

        qkv_cur = self.qkv_proj(hidden_states)
        qkv_cur = op.reshape(qkv_cur, (b, s, h_q + 2 * h_kv, d))
        q, k_cur, v_cur = op.split(qkv_cur, [h_q, h_q + h_kv], axis=2)

        k, v = self.interleave_kv(k_cur, v_cur, kv_seq_len, rolling_cache_len, cache_offset)

        q, k = self.rotary_embedding(q, k, rolling_cache_len)

        if h_kv != h_q:
            k = k.repeat(h_q // h_kv, axis=2)
            v = v.repeat(h_q // h_kv, axis=2)
        q = q.permute_dims([0, 2, 1, 3])  # [b, h, s, d]
        k = k.permute_dims([0, 2, 1, 3])  # [b, h, t, d]
        v = v.permute_dims([0, 2, 1, 3])  # [b, h, t, d]
        attn_weights = op.matmul(
            q, k.permute_dims([0, 1, 3, 2])  # [b, h, s, d] x [b, h, d, t] = [b, h, s, t]
        ) / math.sqrt(d)
        dtype = attn_weights.dtype
        attn_weights = attn_weights.maximum(tir.min_value(dtype)).minimum(attention_mask)
        if dtype == "float32":
            attn_weights = op.softmax(attn_weights, axis=-1)
        else:
            attn_weights = op.softmax(attn_weights.astype("float32"), axis=-1).astype(dtype)
        # [b, h, s, t] x [b, h, t, d] => [b, h, s, d] => [b, s, h, d]
        output = op.matmul(attn_weights, v)
        return self.o_proj(output.permute_dims([0, 2, 1, 3]).reshape((b, s, h_q * d)))


class RollingKVCacheWithSinks(nn.KVCache):
    """
    Rolling buffer cache implementation.
    """

    cache: Optional[rx.Var]

    def override(self, new_element: Tensor, max_cache_size: int, attention_sink_size: int) -> None:
        """
        Override cache elements in RollingKVCacheWithSinks.

        Parameters
        ----------
        new_element : Tensor
            The new tensor to append.

        max_cache_size : int
            Max size of the cache.

        attention_sink_size : int
            Number of stored attention sinks.
        """
        if new_element.dtype != self.dtype:
            raise TypeError(
                f'RollingKVCacheWithSinks has been set to use dtype "{self.dtype}", '
                f'but got "{new_element.dtype}"'
            )
        self.cache = rx.BlockBuilder.current().emit(
            rx.Call(
                rx.extern("vm.builtin.attention_kv_cache_window_override_with_sinks"),
                args=[
                    self.cache,
                    new_element._expr,  # pylint: disable=protected-access
                    rx.PrimValue(max_cache_size),
                    rx.PrimValue(attention_sink_size),
                ],
                sinfo_args=[rx.ObjectStructInfo()],
            )
        )


class MistralDecoderLayer(nn.Module):
    """Exact same as LlamaDecoderLayer."""

    def __init__(self, config: MistralConfig, rotary_embedding: RotaryEmbedding):
        rms_norm_eps = config.rms_norm_eps
        self.self_attn = MistralAttention(config, rotary_embedding)
        self.mlp = MistralMLP(config) if config.num_experts == 0 else MistralMoE(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)
        def _set_tp():
            def _set(layer, hint):
                layer.weight.attrs["shard_strategy"] = hint

            h = config.hidden_size
            hd = config.head_dim
            q = self.self_attn.num_q_heads * hd
            k = self.self_attn.num_kv_heads * hd
            v = self.self_attn.num_kv_heads * hd
            i = self.mlp.intermediate_size
            _set(self.self_attn.qkv_proj, tp.RowSeg("_shard_qkv", rows=[q, k, v], col=h, groups=hd))
            _set(self.self_attn.o_proj, tp.Shard1Dim("_shard_o", shape=self.self_attn.o_proj.weight.shape, axis=1))
            _set(self.mlp.gate_up_proj, tp.RowSeg("_shard_mlp_up", rows=[i, i], col=h, groups=1))
            _set(self.mlp.down_proj, tp.Shard1Dim("_shard_mlp_down", shape=self.mlp.down_proj.weight.shape, axis=1))
        self.tensor_parallel_shards = config.tensor_parallel_shards

    def forward(  # pylint: disable=too-many-arguments
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        rolling_cache_len: tir.Var,
        kv_seq_len: tir.Var,
        cache_offset: tir.Var,
    ):
        """Forward pass of a decoder layer; calculate attention, and add an residual connection."""

        def _apply_residual(out, residual):
            # pylint: disable=no-member
            if self.tensor_parallel_shards > 1:
                return op.ccl_allreduce(out + residual / self.tensor_parallel_shards, "sum")
            # pylint: enable=no-member
            return out + residual

        out = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask,
            rolling_cache_len,
            kv_seq_len,
            cache_offset,
        )
        hidden_states = _apply_residual(out, residual=hidden_states)
        out = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = _apply_residual(out, residual=hidden_states)
        return hidden_states


class MistralModel(nn.Module):
    """Exact same as LlamaModel."""

    def __init__(self, config: MistralConfig):
        assert config.hidden_size % config.num_attention_heads == 0
        rotary_embedding = RotaryEmbedding(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [MistralDecoderLayer(config, rotary_embedding) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.tensor_parallel_shards = config.tensor_parallel_shards > 1

    def forward(  # pylint: disable=too-many-arguments
        self,
        inputs: Tensor,
        rolling_cache_len: tir.Var,
        kv_seq_len: tir.Var,
        cache_offset: tir.Var,
        attention_mask: Tensor,
    ):
        """Forward pass of the model, passing through all decoder layers."""
        # pylint: disable=no-member
        if self.tensor_parallel_shards > 1:
            inputs = op.ccl_broadcast_from_worker0(inputs)
        # pylint: enable=no-member
        hidden_states = self.embed_tokens(inputs)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask, rolling_cache_len, kv_seq_len, cache_offset
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class MistralForCasualLM(nn.Module):
    """Same as LlamaForCausalLM, except for the use of sliding window attention."""

    def __init__(self, config: MistralConfig):
        self.model = MistralModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
        self.sliding_window_size = config.sliding_window_size
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def forward(  # pylint: disable=too-many-arguments
        self,
        inputs: Tensor,
        rolling_cache_len: tir.Var,
        kv_seq_len: tir.Var,
        cache_offset: tir.Var,
        attention_mask: Tensor,
    ):
        """Forward pass."""

        def _index(x: te.Tensor):  # x[:-1,:]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.model(
            inputs, rolling_cache_len, kv_seq_len, cache_offset, attention_mask
        )
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def prefill(
        self,
        inputs: Tensor,
        rolling_cache_len: tir.Var,
        kv_seq_len: tir.Var,
        cache_offset: tir.Var,
    ):
        """
        Prefilling the prompt.

        Parameters
        ----------
        inputs: Tensor
            Input tokens, having ``seq_len`` number of tokens.

        rolling_cache_len: tir.Var
            Number of elements currently in the cache.

        kv_seq_len: tir.Var
            Equals to ``seq_len + rolling_cache_len``.

        cache_offset: tir.Var
            Next position to be overrided on the rolling kv cache.
        """

        def _sliding_window_attention_mask(
            batch_size, seq_len, rolling_cache_len, kv_seq_len, sliding_window_size
        ):
            # See `tests/legacy-python/test_sliding_window_mask.py` for its behavior
            return te.compute(
                (batch_size, 1, seq_len, kv_seq_len),
                lambda b, _, i, j: tir.Select(
                    tir.all(
                        i + rolling_cache_len >= j, i + rolling_cache_len - j < sliding_window_size
                    ),
                    tir.max_value(self.dtype),
                    tir.min_value(self.dtype),
                ),
                name="sliding_window_attention_mask_prefill",
            )

        batch_size, seq_len = inputs.shape
        attention_mask = op.tensor_expr_op(
            _sliding_window_attention_mask,
            name_hint="sliding_window_attention_mask_prefill",
            args=[
                batch_size,
                seq_len,
                rolling_cache_len,
                kv_seq_len,
                self.sliding_window_size,
            ],
        )
        return self.forward(inputs, rolling_cache_len, kv_seq_len, cache_offset, attention_mask)

    def decode(
        self,
        inputs: Tensor,
        rolling_cache_len: tir.Var,
        kv_seq_len: tir.Var,
        cache_offset: tir.Var,
    ):
        """Decoding step."""
        batch_size, seq_len = inputs.shape
        attention_mask = op.full(
            shape=[batch_size, 1, seq_len, kv_seq_len],
            fill_value=tir.max_value(self.dtype),
            dtype=self.dtype,
        )
        return self.forward(inputs, rolling_cache_len, kv_seq_len, cache_offset, attention_mask)

    def softmax_with_temperature(self, logits: Tensor, temperature: Tensor):
        """Softmax."""
        return op.softmax(logits / temperature, axis=-1)

    def get_default_spec(self):
        """Needed for ``export_tvm()``."""
        batch_size = 1
        mod_spec = {
            "prefill": {
                "inputs": nn.spec.Tensor([batch_size, "seq_len"], "int32"),
                "rolling_cache_len": int,
                "kv_seq_len": int,
                "cache_offset": int,
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "packed",
                },
            },
            "decode": {
                "inputs": nn.spec.Tensor([batch_size, 1], "int32"),
                "rolling_cache_len": int,
                "kv_seq_len": int,
                "cache_offset": int,
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "packed",
                },
            },
            "softmax_with_temperature": {
                "logits": nn.spec.Tensor([1, 1, "vocab_size"], "float32"),
                "temperature": nn.spec.Tensor([], "float32"),
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
