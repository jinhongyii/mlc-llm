"""Mixture of Experts operators"""
import tvm
from tvm import DataType, te, tir
from tvm.relax.frontend.nn import Tensor, op
from tvm.script import tir as T
from tvm.topi.cuda.sort import topk as topi_topk

from mlc_chat.support import logging

logger = logging.getLogger(__name__)

# pylint: skip-file
# mypy: ignore-errors


def get_indptr(cumsum_colwise_flattened: Tensor, num_local_experts: int) -> Tensor:
    @T.prim_func
    def get_expert_instance_indptr(
        var_cumsum_colwise_flattened: T.handle,
        var_expert_instance_indptr: T.handle,
        batch_size: T.int32,
    ):
        cumsum_colwise_flattened = T.match_buffer(
            var_cumsum_colwise_flattened, shape=[batch_size * num_local_experts], dtype="int32"
        )
        expert_instance_indptr = T.match_buffer(
            var_expert_instance_indptr, shape=[num_local_experts + 1], dtype="int32"
        )

        for expert_id in T.serial(0, num_local_experts + 1):
            with T.block("indptr"):
                vexpert_id = T.axis.spatial(num_local_experts + 1, expert_id)
                expert_instance_indptr[vexpert_id] = T.Select(
                    condition=vexpert_id > 0,
                    true_value=cumsum_colwise_flattened[vexpert_id * batch_size - 1],
                    false_value=T.int32(0),
                )

    return op.tensor_ir_op(
        get_expert_instance_indptr,
        "get_expert_instance_indptr",
        args=[cumsum_colwise_flattened, cumsum_colwise_flattened.shape[0] // num_local_experts],
        out=Tensor.placeholder([num_local_experts + 1], "int32"),
    )


def get_indices(
    cumsum_colwise_flattened: Tensor, expert_indices: Tensor, num_experts_per_tok: int
) -> Tensor:
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
            var_expert_indices, shape=[batch_size, num_experts_per_tok], dtype="int32"
        )
        flattened_expert_indices = T.match_buffer(
            var_flattened_expert_indices,
            shape=[batch_size * num_experts_per_tok],
            dtype="int32",
        )

        for io in T.thread_binding(
            0, T.floordiv(cumsum_flattened_length, T.int32(1024)), "blockIdx.x"
        ):
            for ii in T.thread_binding(0, T.int32(1024), "threadIdx.x"):
                with T.block("get_indices"):
                    vi = T.axis.spatial(cumsum_flattened_length, io * T.int32(1024) + ii)
                    T.where(io * T.int32(1024) + ii < cumsum_flattened_length)
                    T.reads(cumsum_colwise_flattened[vi - 1 : vi - 1 + 2], expert_indices[:, 0:2])
                    T.writes(flattened_expert_indices[:])
                    expert_idx = T.alloc_buffer(shape=(), dtype="int32", scope="local")
                    if (vi == 0 and cumsum_colwise_flattened[vi] > 0) or cumsum_colwise_flattened[
                        vi
                    ] != cumsum_colwise_flattened[vi - 1]:
                        idx: T.SizeVar("idx", "int32") = cumsum_colwise_flattened[vi] - 1
                        instance_id: T.SizeVar("instance_id", "int32") = T.truncmod(vi, batch_size)
                        expert_id: T.SizeVar("expert_id", "int32") = T.truncdiv(vi, batch_size)
                        for j in T.serial(0, num_experts_per_tok):
                            with T.block("select_expert"):
                                vj = T.axis.spatial(num_experts_per_tok, j)
                                vinstance_id = T.axis.spatial(batch_size, instance_id)
                                vexpert_id = T.axis.spatial(
                                    T.truncdiv(cumsum_flattened_length, batch_size), expert_id
                                )
                                if expert_indices[vinstance_id, vj] == vexpert_id:
                                    expert_idx[()] = vj
                        flattened_expert_indices[idx] = (
                            instance_id * num_experts_per_tok + expert_idx[()]
                        )

    return op.tensor_ir_op(
        get_flattened_expert_indices_scheduled,
        "get_flattened_expert_indices",
        args=[cumsum_colwise_flattened, expert_indices],
        out=Tensor.placeholder([expert_indices.shape[0] * num_experts_per_tok], "int32"),
    )


def scatter_output(flattened_indices: Tensor, linear_out: Tensor, dtype: str) -> Tensor:
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
            dtype=dtype,
        )
        flattened_expert_indices = T.match_buffer(
            var_flattened_expert_indices, shape=[flattened_indices_length], dtype="int32"
        )
        scattered_output = T.match_buffer(
            var_scattered_output,
            shape=[flattened_indices_length, out_features],
            dtype=dtype,
        )

        for i in T.serial(0, flattened_indices_length):
            for j in T.serial(0, out_features):
                with T.block("scatter"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    scattered_output[flattened_expert_indices[vi], vj] = unscattered_output[vi, vj]

    return op.tensor_ir_op(
        tir_scatter_output,
        "scatter_output",
        args=[linear_out, flattened_indices],
        out=Tensor.placeholder(linear_out.shape, linear_out.dtype),
    )


def topk(x: Tensor, k: int, num_local_experts: int, dtype: str, index_dtype: str):
    # specialized kernel for top 2 case
    @T.prim_func
    def top2_func(
        x_handle: T.handle,
        out_handle: T.handle,
        out_index_handle: T.handle,
    ) -> None:
        total_rows = T.int64()
        x = T.match_buffer(x_handle, (total_rows, num_local_experts), dtype)
        out = T.match_buffer(out_handle, (total_rows, 2), dtype)
        out_index = T.match_buffer(out_index_handle, (total_rows, 2), index_dtype)
        local_top_k = T.alloc_buffer((2,), dtype=dtype, scope="local")
        local_top_k_index = T.alloc_buffer((2,), dtype=index_dtype, scope="local")
        T.func_attr({"tir.noalias": True, "tir.is_scheduled": True})
        for io in T.thread_binding(0, T.ceildiv(total_rows, T.int64(1024)), "blockIdx.x"):
            for ii in T.thread_binding(0, T.min(total_rows, T.int64(1024)), "threadIdx.x"):
                with T.block("top2"):
                    vi = T.axis.spatial(total_rows, io * T.int64(1024) + ii)
                    T.where(io * T.int64(1024) + ii < total_rows)
                    with T.block("init"):
                        local_top_k[0] = T.min_value(dtype)
                        local_top_k_index[0] = 0
                    for k in range(num_local_experts):
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
        return op.tensor_expr_op(topi_topk, "topk", args=[x, k, -1, "both", False, "int32"])

    return op.tensor_ir_op(
        top2_func,
        "top2",
        args=[x],
        out=(
            Tensor.placeholder([x.shape[0], 2], dtype),
            Tensor.placeholder([x.shape[0], 2], index_dtype),
        ),
    )


def topk_mask(topk_indices: Tensor, num_experts_per_tok: int, num_local_experts: int) -> Tensor:
    from functools import reduce

    def te_topk_mask_op(topk_indices):
        ntokens = topk_indices.shape[0]
        assert topk_indices.shape[1] == num_experts_per_tok
        return te.compute(
            (ntokens, num_local_experts),
            lambda i, j: tir.expr.Select(
                reduce(
                    lambda a, b: tir.Or(a, b),
                    [topk_indices[i, k] == j for k in range(num_experts_per_tok)],
                ),
                true_value=tir.const(1, "int32"),
                false_value=tir.const(0, "int32"),
            ),
        )

    return op.tensor_expr_op(te_topk_mask_op, "topk_mask", args=[topk_indices])


def gemv_e1_e3(
    x: Tensor,
    weight: Tensor,
    indptr: Tensor,
    in_features: int,
    out_features: int,
    num_experts_per_tok: int,
    num_local_experts: int,
    dtype: str,
):
    # fmt: off

    @T.prim_func
    def _gemv_e1_e3(var_x: T.handle, var_w: T.handle, var_indptr: T.handle, var_o: T.handle):
        T.func_attr({"op_pattern": 4})
        x = T.match_buffer(var_x, (1, in_features), dtype)
        w = T.match_buffer(var_w, (num_local_experts, out_features, in_features), dtype)
        indptr = T.match_buffer(var_indptr, (num_experts_per_tok,), "int32")
        o = T.match_buffer(var_o, (num_experts_per_tok, out_features), dtype)
        # with T.block("root"):
        for expert_id in T.thread_binding(num_experts_per_tok, thread="blockIdx.y"):
            with T.block("gemv_o"):
                v_expert_id_o = T.axis.spatial(num_experts_per_tok, expert_id)
                vi_o = T.axis.spatial(1, 0)
                vj_o = T.axis.reduce(1, 0)
                T.reads(x[0, 0:in_features], w[indptr[v_expert_id_o], 0:out_features, 0:in_features], indptr[v_expert_id_o])
                T.writes(o[v_expert_id_o, 0:out_features])
                for i, j in T.grid(out_features, in_features):
                    with T.block("gemv"):
                        vi_i, vj_i = T.axis.remap("SR", [i, j])
                        T.reads(x[0, vj_i], w[indptr[v_expert_id_o], vi_i, vj_i], indptr[v_expert_id_o])
                        T.writes(o[v_expert_id_o, vi_i])
                        with T.init():
                            o[v_expert_id_o, vi_i] = T.cast(T.float16(0), dtype)
                        o[v_expert_id_o, vi_i] = o[v_expert_id_o, vi_i] + x[0, vj_i] * w[indptr[v_expert_id_o], vi_i, vj_i]
    # fmt: on
    return op.tensor_ir_op(
        _gemv_e1_e3,
        "gemv_e1_e3",
        args=[x, weight, indptr],
        out=Tensor.placeholder([indptr.shape[0], out_features], dtype),
    )


def gemv_e2(
    x: Tensor,
    weight: Tensor,
    indptr: Tensor,
    in_features: int,
    out_features: int,
    num_experts_per_tok: int,
    num_local_experts: int,
    dtype: str,
):
    # fmt: off

    @T.prim_func
    def _gemv_e2(var_x: T.handle, var_w: T.handle, var_indptr: T.handle, var_o: T.handle):
        T.func_attr({"op_pattern": 4})
        x = T.match_buffer(var_x, (num_experts_per_tok, in_features), dtype)
        w = T.match_buffer(var_w, (num_local_experts, out_features, in_features), dtype)
        indptr = T.match_buffer(var_indptr, (num_experts_per_tok,), "int32")
        o = T.match_buffer(var_o, (num_experts_per_tok, out_features), dtype)
        # with T.block("root"):
        for expert_id in T.thread_binding(num_experts_per_tok, thread="blockIdx.y"):
            with T.block("gemv_o"):
                v_expert_id_o = T.axis.spatial(num_experts_per_tok, expert_id)
                vi_o = T.axis.spatial(1, 0)
                vj_o = T.axis.reduce(1, 0)
                T.reads(x[v_expert_id_o, 0:in_features], w[indptr[v_expert_id_o], 0:out_features, 0:in_features], indptr[v_expert_id_o])
                T.writes(o[v_expert_id_o, 0:out_features])
                for i, j in T.grid(out_features, in_features):
                    with T.block("gemv"):
                        vi_i, vj_i = T.axis.remap("SR", [i, j])
                        T.reads(x[v_expert_id_o, vj_i], w[indptr[v_expert_id_o], vi_i, vj_i], indptr[v_expert_id_o])
                        T.writes(o[v_expert_id_o, vi_i])
                        with T.init():
                            o[v_expert_id_o, vi_i] = T.cast(T.float16(0), dtype)
                        o[v_expert_id_o, vi_i] = o[v_expert_id_o, vi_i] + x[v_expert_id_o, vj_i] * w[indptr[v_expert_id_o], vi_i, vj_i]
    # fmt: on

    return op.tensor_ir_op(
        _gemv_e2,
        "gemv_e2",
        args=[x, weight, indptr],
        out=Tensor.placeholder([indptr.shape[0], out_features], dtype),
    )


def group_gemm(
    input: Tensor,
    weight: Tensor,
    indptr: Tensor,
    in_features: int,
    out_features: int,
    num_local_experts: int,
    dtype: str,
):
    Ne = num_local_experts
    N = out_features
    K = in_features

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
        weight: T.Buffer((Ne, N, K), dtype=dtype),
        indptr: T.Buffer((Ne + 1), dtype="int32"),
        var_O: T.handle,
    ):
        T.func_attr({"tir.is_scheduled": 1})
        B = T.int32(is_size_var=True)
        X = T.match_buffer(var_X, (B, K), dtype)
        O = T.match_buffer(var_O, (B, N), dtype)

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
                            X_tile = T.alloc_buffer((BLK_M, K), dtype, scope="shared")
                            W_tile = T.alloc_buffer((BLK_N, K), dtype, scope="shared")
                            O_tile = T.alloc_buffer((BLK_M, BLK_N), "float32", scope="local")
                            
                            for a0, a1 in T.grid(BLK_M, K): 
                                with T.block("X_shared"):
                                    i, j = T.axis.remap("SS", [a0, a1])
                                    X_tile[i, j] = T.if_then_else(tile_m_start + i < row[1], X[tile_m_start + i, j], tir.const(0, dtype))
                            for a0, a1 in T.grid(BLK_N, K):
                                with T.block("W_shared"):
                                    i, j = T.axis.remap("SS", [a0, a1])
                                    n: T.int32 = tile_n_start + i
                                    W_tile[i, j] = T.if_then_else(
                                        n < N, 
                                        weight[a, n, j],
                                        tir.const(0, dtype)
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

    sch = tir.Schedule(group_gemm)

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

    return op.tensor_ir_op(
        func,
        "group_gemm",
        args=[input, weight, indptr],
        out=Tensor.placeholder([input.shape[0], out_features], dtype),
    )


def group_dequantize_gemv_e1_e3(
    x: Tensor,
    w: Tensor,
    scale: Tensor,
    indptr: Tensor,
    config: "GroupQuantize",
    in_features: int,
    out_features: int,
    num_experts_per_tok: int,
    num_local_experts: int,
):
    bits = DataType(config.quantize_dtype).bits
    tir_max_int = tir.const(config.max_int_value, config.model_dtype)
    # fmt: off

    @T.prim_func
    def dequantize_gemv_e1_e3(var_x: T.handle, var_w: T.handle, var_scale:T.handle, var_indptr: T.handle, var_o: T.handle):
        T.func_attr({"op_pattern": 4})
        x = T.match_buffer(var_x, (1, in_features), config.model_dtype)
        w = T.match_buffer(var_w, (num_local_experts, out_features, in_features //config.num_elem_per_storage), config.storage_dtype)
        scale = T.match_buffer(var_scale, (num_local_experts, out_features, in_features//config.group_size), config.model_dtype)
        indptr = T.match_buffer(var_indptr, (num_experts_per_tok,), "int32")
        o = T.match_buffer(var_o, (num_experts_per_tok, out_features), config.model_dtype)
        # with T.block("root"):
        for expert_id in T.thread_binding(num_experts_per_tok, thread="blockIdx.y"):
            with T.block("gemv_o"):
                v_expert_id_o = T.axis.spatial(num_experts_per_tok, expert_id)
                vi_o = T.axis.spatial(1, 0)
                vj_o = T.axis.reduce(1, 0)
                compute = T.alloc_buffer((out_features, in_features), config.model_dtype)
                dequantize = T.alloc_buffer((out_features, in_features), config.model_dtype)
                for i1, i2 in T.grid(out_features, in_features):
                    with T.block("compute"):
                        v_i1, v_i2 = T.axis.remap("SS", [i1, i2])
                        compute[v_i1, v_i2] = T.Cast(config.model_dtype, T.bitwise_and(T.shift_right(w[indptr[v_expert_id_o], v_i1, v_i2 // config.num_elem_per_storage],
                                                                                                T.Cast(config.storage_dtype, v_i2 % config.num_elem_per_storage * bits)),
                                                                                tir.const((1 << bits) - 1, config.storage_dtype)))
                for i1, i2 in T.grid(out_features, in_features):
                    with T.block("dequantize"):
                        v_i1, v_i2 = T.axis.remap("SS", [i1, i2])
                        dequantize[v_i1, v_i2] = (compute[v_i1, v_i2] - tir_max_int) * scale[indptr[v_expert_id_o], v_i1, v_i2 // config.group_size]
                for i, j in T.grid(out_features, in_features):
                    with T.block("gemv"):
                        vi_i, vj_i = T.axis.remap("SR", [i, j])
                        T.reads(x[0, vj_i], dequantize[vi_i, vj_i], indptr[v_expert_id_o])
                        T.writes(o[v_expert_id_o, vi_i])
                        with T.init():
                            o[v_expert_id_o, vi_i] = T.cast(T.float16(0), config.model_dtype)
                        o[v_expert_id_o, vi_i] = o[v_expert_id_o, vi_i] + x[0, vj_i] * dequantize[vi_i, vj_i]
    # fmt: on

    return op.tensor_ir_op(
        dequantize_gemv_e1_e3,
        "dequantize_gemv_e1_e3",
        args=[x, w, scale, indptr],
        out=Tensor.placeholder([indptr.shape[0], out_features], config.model_dtype),
    )


def group_dequantize_gemv_e2(
    x: Tensor,
    w: Tensor,
    scale: Tensor,
    indptr: Tensor,
    config: "GroupQuantize",
    in_features: int,
    out_features: int,
    num_experts_per_tok: int,
    num_local_experts: int,
):
    bits = DataType(config.quantize_dtype).bits
    tir_max_int = tir.const(config.max_int_value, config.model_dtype)

    # fmt: off

    @T.prim_func
    def dequantize_gemv_e2(var_x: T.handle, var_w: T.handle, var_scale:T.handle, var_indptr: T.handle, var_o: T.handle):
        T.func_attr({"op_pattern": 4})
        x = T.match_buffer(var_x, (num_experts_per_tok, in_features), config.model_dtype)
        w = T.match_buffer(var_w, (num_local_experts, out_features, in_features //config.num_elem_per_storage), config.storage_dtype)
        scale = T.match_buffer(var_scale, (num_local_experts, out_features, in_features//config.group_size), config.model_dtype)
        indptr = T.match_buffer(var_indptr, (num_experts_per_tok, ), "int32")
        o = T.match_buffer(var_o, (num_experts_per_tok, out_features), config.model_dtype)
        # with T.block("root"):
        for expert_id in T.thread_binding(num_experts_per_tok, thread="blockIdx.y"):
            with T.block("gemv_o"):
                v_expert_id_o = T.axis.spatial(num_experts_per_tok, expert_id)
                vi_o = T.axis.spatial(1, 0)
                vj_o = T.axis.reduce(1, 0)
                compute = T.alloc_buffer((out_features, in_features), config.model_dtype)
                dequantize = T.alloc_buffer((out_features, in_features), config.model_dtype)
                for i1, i2 in T.grid(out_features, in_features):
                    with T.block("compute"):
                        v_i1, v_i2 = T.axis.remap("SS", [i1, i2])
                        compute[v_i1, v_i2] = T.Cast(config.model_dtype, T.bitwise_and(T.shift_right(w[indptr[v_expert_id_o], v_i1, v_i2 // config.num_elem_per_storage],
                                                                                                T.Cast(config.storage_dtype, v_i2 % config.num_elem_per_storage * bits)),
                                                                                tir.const((1 << bits) - 1, config.storage_dtype)))
                for i1, i2 in T.grid(out_features, in_features):
                    with T.block("dequantize"):
                        v_i1, v_i2 = T.axis.remap("SS", [i1, i2])
                        dequantize[v_i1, v_i2] = (compute[v_i1, v_i2] - tir_max_int) * scale[indptr[v_expert_id_o], v_i1, v_i2 // config.group_size]
                for i, j in T.grid(out_features, in_features):
                    with T.block("gemv"):
                        vi_i, vj_i = T.axis.remap("SR", [i, j])
                        with T.init():
                            o[v_expert_id_o, vi_i] = T.cast(T.float16(0), config.model_dtype)
                        o[v_expert_id_o, vi_i] = o[v_expert_id_o, vi_i] + x[v_expert_id_o, vj_i] * dequantize[vi_i, vj_i]

    # fmt: on

    return op.tensor_ir_op(
        dequantize_gemv_e2,
        "dequantize_gemv_e2",
        args=[x, w, scale, indptr],
        out=Tensor.placeholder([indptr.shape[0], out_features], config.model_dtype),
    )


# currently highly optimized for metal
def group_dequantize_group_gemm(
    input: Tensor,
    weight: Tensor,
    scale: Tensor,
    indptr: Tensor,
    config: "GroupQuantize",
    in_features: int,
    out_features: int,
    num_local_experts: int,
):
    Ne = num_local_experts
    N = out_features
    K = in_features
    bits = DataType(config.quantize_dtype).bits
    tir_max_int = tir.const(config.max_int_value, config.model_dtype)
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
    def dequantize_group_gemm(
        var_X: T.handle,
        weight_0: T.Buffer((Ne, N, K // config.num_elem_per_storage), dtype=config.storage_dtype),
        weight_1: T.Buffer((Ne, N, K // config.group_size), dtype=config.model_dtype),
        indptr: T.Buffer((Ne + 1), dtype="int32"),
        var_O: T.handle,
    ):
        T.func_attr({"tir.is_scheduled": 1})
        B = T.int32(is_size_var=True)
        X = T.match_buffer(var_X, (B, K), config.model_dtype)
        O = T.match_buffer(var_O, (B, N), config.model_dtype)

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
                            X_tile = T.alloc_buffer((BLK_M, K), config.model_dtype, scope="shared")
                            W_tile = T.alloc_buffer((BLK_N, K), config.model_dtype, scope="shared")
                            O_tile = T.alloc_buffer((BLK_M, BLK_N), "float32", scope="local")
                            
                            for a0, a1 in T.grid(BLK_M, K): 
                                with T.block("X_shared"):
                                    i, j = T.axis.remap("SS", [a0, a1])
                                    X_tile[i, j] = T.if_then_else(tile_m_start + i < row[1], X[tile_m_start + i, j], tir.const(0, config.model_dtype))
                            for a0, a1 in T.grid(BLK_N, K):
                                with T.block("W_shared"):
                                    i, j = T.axis.remap("SS", [a0, a1])
                                    n: T.int32 = tile_n_start + i
                                    W_tile[i, j] = T.if_then_else(
                                        n < N, 
                                        (T.Cast(config.model_dtype, T.bitwise_and(T.shift_right(weight_0[a, n, j // config.num_elem_per_storage], T.Cast(config.storage_dtype, j % config.num_elem_per_storage * bits)), tir.const((1 << bits) - 1, config.storage_dtype))) - tir_max_int) * weight_1[a, n, j // config.group_size], 
                                        tir.const(0, config.model_dtype)
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

    sch = tir.Schedule(dequantize_group_gemm)

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
    # FIXME: group gemm will be duplicated in the IR
    return op.tensor_ir_op(
        func,
        "dequantize_group_gemm",
        args=[input, weight, scale, indptr],
        out=Tensor.placeholder([input.shape[0], out_features], config.model_dtype),
    )
