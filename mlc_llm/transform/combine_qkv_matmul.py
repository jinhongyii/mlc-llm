from typing import List, Tuple
from collections import namedtuple

import tvm
from tvm import relax, tir
from tvm.tir.stmt_functor import post_order_visit, ir_transform
from tvm.relax.dpl import (
    PatternContext,
    is_op,
    rewrite_bindings,
    wildcard,
    is_tuple_get_item,
    GlobalVarPattern,
    TuplePattern,
    is_shape,
)
from tvm.script import _ffi_api as _ffi
from tvm.script import relax as R, tir as T


ParamRewriteSpec=namedtuple("ParamRewriteSpec", ["param_index", "dim", "new_value"])
def rewrite_tir(prim_func: tir.PrimFunc, param_rewrite_spec: ParamRewriteSpec) -> (tir.PrimFunc, bool):
    param = prim_func.params[param_rewrite_spec.param_index]
    buffer = prim_func.buffer_map[param]
    block_var = None
    loop_var = None
    success = True
    def fcollect_var(e):
        nonlocal block_var, loop_var, success
        if isinstance(e, (tir.BufferLoad, tir.BufferStore)) and buffer.same_as(e.buffer):
            block_var = e.indices[param_rewrite_spec.dim]
            if not isinstance(block_var, tir.Var):
                success = False
        elif isinstance(e, tir.BlockRealize):
            for (i, value) in enumerate(e.iter_values):
                if e.block.iter_vars[i].var.same_as(block_var):
                    loop_var = value
                    if not isinstance(value, tir.Var):
                        success = False
    
    post_order_visit(prim_func.body, fcollect_var)
    print("loop var:", loop_var)
    if not success:
        return prim_func, False
    
    buffer_axes = []
    def fcollect_buffer(e):
        nonlocal buffer_axes
        if isinstance(e, (tir.BufferLoad, tir.BufferStore)):
            for i, index in enumerate(e.indices):
                if index.same_as(block_var):
                    buffer_axes.append((e.buffer, i))
    post_order_visit(prim_func.body, fcollect_buffer)
    buffer_replace_map = {}
    for buffer, i in buffer_axes:
        new_shape = list(buffer.shape)
        new_shape[i] = param_rewrite_spec.new_value
        new_buffer = tir.decl_buffer(new_shape, buffer.dtype, buffer.name, buffer.data, buffer.strides, buffer.elem_offset, buffer.scope, buffer.data_alignment, buffer.offset_factor)
        buffer_replace_map[buffer] = new_buffer
        
    def postorder(stmt):
        if isinstance(stmt, tir.For) and stmt.loop_var.same_as(loop_var):                    
            return tir.For(stmt.loop_var, stmt.min, param_rewrite_spec.new_value, stmt.kind, stmt.body, stmt.thread_binding, stmt.annotations)
        elif isinstance(stmt, tir.Block):
            new_iter_vars = [tir.IterVar(tvm.ir.Range.from_min_extent(iter_var.dom.min, param_rewrite_spec.new_value), iter_var.var, iter_var.iter_type, iter_var.thread_tag) if iter_var.var.same_as(block_var) else iter_var for iter_var in stmt.iter_vars]
            new_annotations = dict(stmt.annotations)
            new_annotations["tir.script_parsing_detect_access"] = 3
            return tir.Block(new_iter_vars, [], [], stmt.name_hint, stmt.body, stmt.init, stmt.alloc_buffers, stmt.match_buffers, new_annotations)
        elif isinstance(stmt, tir.BufferStore) and stmt.buffer in buffer_replace_map:
            return tir.BufferStore(buffer_replace_map[stmt.buffer], stmt.value, stmt.indices)
        elif isinstance(stmt, tir.BufferLoad) and stmt.buffer in buffer_replace_map:
            return tir.BufferLoad(buffer_replace_map[stmt.buffer], stmt.indices)

    new_body = ir_transform(prim_func.body, None, postorder)
    new_buffer_map = dict(prim_func.buffer_map)
    for var, buffer in new_buffer_map.items():
        if buffer in buffer_replace_map:
            new_buffer_map[var] = buffer_replace_map[buffer]
    new_func = tir.PrimFunc(prim_func.params, new_body, prim_func.ret_type, new_buffer_map, prim_func.attrs)
    new_func = _ffi.Complete(new_func, [])
    param_sinfo = []
    for param in new_func.params:
        if param in new_func.buffer_map:
            buf = new_func.buffer_map[param]
            sinfo = relax.TensorStructInfo(shape=buf.shape, dtype=buf.dtype)
        else:
            sinfo = relax.PrimStructInfo(param.dtype)
        param_sinfo.append(sinfo)

    relax.expr._update_struct_info(
        new_func,
        tvm.relax.FuncStructInfo(
            params=param_sinfo,
            ret=relax.TupleStructInfo([]),
            purity=False,
        ),
    )
    return new_func, True

def combine_qkv():
    @tvm.ir.transform.module_pass(opt_level=0, name="fuse_split_rotary_embedding")
    def ir_module_pass(mod: tvm.IRModule, _pass_context) -> tvm.IRModule:
        with PatternContext() as ctx:
            pat_input = wildcard()
            pat_wq = wildcard()
            pat_wk = wildcard()
            pat_wv = wildcard()
            pat_matmul_gvar = GlobalVarPattern("(NT_)?matmul.*")
            pat_q = is_op("relax.call_tir")(
                pat_matmul_gvar,
                TuplePattern([pat_input, pat_wq]),
                add_constraint=False
            )
            pat_input.used_by(pat_q)
            pat_wq.used_by(pat_q)
            pat_k = is_op("relax.call_tir")(
                pat_matmul_gvar,
                TuplePattern([pat_input, pat_wk]),
                add_constraint=False
            )
            pat_input.used_by(pat_k)
            pat_wk.used_by(pat_k)
            pat_v = is_op("relax.call_tir")(
                pat_matmul_gvar,
                TuplePattern([pat_input, pat_wv]),
                add_constraint=False
            )
            pat_wv.used_by(pat_v)
            pat_input.used_by(pat_v)

        def rewriter(matchings, bindings):
            # Extracting all the relax and TIR variables that we'll need

            input = matchings[pat_input]

            wq = matchings[pat_wq]
            wk = matchings[pat_wk]
            wv = matchings[pat_wv]
            q = matchings[pat_q]
            k = matchings[pat_k]
            v = matchings[pat_v]
            
            matmul_gvar = bindings[q].args[0]
            assert matmul_gvar.name_hint.startswith("matmul") or matmul_gvar.name_hint.startswith("NT_matmul")
            decode_gvar = bindings[wq].args[0]
            assert decode_gvar.name_hint.startswith("decode")
            
            wq_data, wq_scale = bindings[wq].args[1]
            wq_data_can_lift, wq_scale_can_lift = bindings[wq_data].args[0], bindings[wq_scale].args[0]
            wk_data, wk_scale = bindings[wk].args[1]
            wk_data_can_lift, wk_scale_can_lift = bindings[wk_data].args[0], bindings[wk_scale].args[0]
            wv_data, wv_scale = bindings[wv].args[1]
            wv_data_can_lift, wv_scale_can_lift = bindings[wv_data].args[0], bindings[wv_scale].args[0]
            
            concat_wqkv_data_can_lift = R.concat([wq_data_can_lift, wk_data_can_lift, wv_data_can_lift], axis=0)
            concat_wqkv_scale_can_lift = R.concat([wq_scale_can_lift, wk_scale_can_lift, wv_scale_can_lift], axis=0)
            concat_wqkv_data = R.stop_lift_params(concat_wqkv_data_can_lift)
            concat_wqkv_scale = R.stop_lift_params(concat_wqkv_scale_can_lift)
            
            decode_combine_func_name = decode_gvar.name_hint + "_combine_qkv"
            matmul_combine_func_name = matmul_gvar.name_hint + "_combine_qkv"
            if not any(gvar.name_hint == decode_combine_func_name for gvar in mod.functions ):
                decode_tir = mod[decode_gvar]
                new_decode_tir, decode_rewrite_success = rewrite_tir(decode_tir, ParamRewriteSpec(param_index=0, dim=0, new_value=wq.struct_info.shape[0] + wk.struct_info.shape[0] + wv.struct_info.shape[0]))
                assert decode_rewrite_success
                mod[decode_combine_func_name] = new_decode_tir
                new_decode_gvar = mod.get_global_var(decode_combine_func_name)
                relax.expr._update_struct_info(new_decode_gvar, mod[decode_combine_func_name].struct_info)
            else:
                new_decode_gvar = mod.get_global_var(decode_combine_func_name)

            if not any(gvar.name_hint == matmul_combine_func_name for gvar in mod.functions ):
                matmul_tir = mod[matmul_gvar]
                new_matmul_tir, matmul_rewrite_success = rewrite_tir(matmul_tir, ParamRewriteSpec(param_index=1, dim=0, new_value=wq.struct_info.shape[0] + wk.struct_info.shape[0] + wv.struct_info.shape[0]))
                assert matmul_rewrite_success
                mod[matmul_combine_func_name] = new_matmul_tir
                new_matmul_gvar = mod.get_global_var(matmul_combine_func_name)
                relax.expr._update_struct_info(new_matmul_gvar, mod[matmul_combine_func_name].struct_info)
            else:
                new_matmul_gvar = mod.get_global_var(matmul_combine_func_name)
                
            concat_wqkv_shape = list(wq.struct_info.shape)
            concat_wqkv_shape[0] = wq.struct_info.shape[0] + wk.struct_info.shape[0] + wv.struct_info.shape[0]
            concat_wqkv_sinfo = R.Tensor(concat_wqkv_shape, dtype="float16")
            concat_wqkv = R.call_tir(
                new_decode_gvar,
                (concat_wqkv_data, concat_wqkv_scale),
                out_sinfo=concat_wqkv_sinfo
            )
            concat_qkv_shape = list(q.struct_info.shape)
            concat_qkv_shape[-1] = q.struct_info.shape[-1] + k.struct_info.shape[-1] + v.struct_info.shape[-1]
            concat_qkv_sinfo = R.Tensor(concat_qkv_shape, dtype="float16")
            concat_qkv = R.call_tir(
                new_matmul_gvar,
                (input, concat_wqkv),
                out_sinfo=concat_qkv_sinfo
            )
            qkv_tuple = R.split(concat_qkv, [q.struct_info.shape[-1], q.struct_info.shape[-1] + k.struct_info.shape[-1]], axis=-1)
            new_q = R.TupleGetItem(qkv_tuple, 0)
            new_k = R.TupleGetItem(qkv_tuple, 1)
            new_v = R.TupleGetItem(qkv_tuple, 2)
            
            return {
                q: new_q,
                k: new_k,
                v: new_v,
            }

        new_mod = {}
        for gvar, func in mod.functions.items():
            if isinstance(func, relax.Function):
                func = rewrite_bindings(ctx, rewriter, func)
            new_mod[gvar] = func
            
        for gvar, func in mod.functions.items():
            if isinstance(func, tir.PrimFunc) and gvar not in new_mod:
                new_mod[gvar] = func

        new_mod = tvm.IRModule(new_mod, mod.type_definitions, mod.attrs, mod.global_infos)
        return new_mod

    return ir_module_pass
