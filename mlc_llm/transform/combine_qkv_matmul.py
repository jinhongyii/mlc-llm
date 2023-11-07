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


def combine_parallel_matmul(mod, func, num_parallel):
    with PatternContext() as ctx:
        pat_input = wildcard()
        pat_weights = [wildcard() for _ in range(num_parallel)]
        pat_matmul_gvar = GlobalVarPattern("(NT_)?matmul.*")
        pat_matmul_outputs = []
        for pat_w in pat_weights:
            pat_o = is_op("relax.call_tir")(
                pat_matmul_gvar,
                TuplePattern([pat_input, pat_w]),
                add_constraint=False
            )
            pat_matmul_outputs.append(pat_o)
            pat_input.used_by(pat_o)
            pat_w.used_by(pat_o)
            
    def rewriter(matchings, bindings):
        # Extracting all the relax and TIR variables that we'll need

        input = matchings[pat_input]
        weights = [matchings[pat_w] for pat_w in pat_weights]
        matmul_outputs = [matchings[pat_o] for pat_o in pat_matmul_outputs]
        
        matmul_gvar = bindings[matmul_outputs[0]].args[0]
        assert matmul_gvar.name_hint.startswith("matmul") or matmul_gvar.name_hint.startswith("NT_matmul")
        decode_gvar = bindings[weights[0]].args[0]
        assert decode_gvar.name_hint.startswith("decode")
        
        weights_data_to_lift = []
        weights_scale_to_lift = []
        for w in weights:
            w_data, w_scale = bindings[w].args[1]
            w_data_to_lift, w_scale_to_lift = bindings[w_data].args[0], bindings[w_scale].args[0]
            weights_data_to_lift.append(w_data_to_lift)
            weights_scale_to_lift.append(w_scale_to_lift)
        
        concat_wqkv_data_can_lift = R.concat(weights_data_to_lift, axis=0)
        concat_wqkv_scale_can_lift = R.concat(weights_scale_to_lift, axis=0)
        concat_wqkv_data = R.stop_lift_params(concat_wqkv_data_can_lift)
        concat_wqkv_scale = R.stop_lift_params(concat_wqkv_scale_can_lift)
        
        decode_combine_func_name = decode_gvar.name_hint + "_combine_matmul"
        matmul_combine_func_name = matmul_gvar.name_hint + "_combine_matmul"
        if not any(gvar.name_hint == decode_combine_func_name for gvar in mod.functions ):
            decode_tir = mod[decode_gvar]
            new_decode_tir, decode_rewrite_success = rewrite_tir(decode_tir, ParamRewriteSpec(param_index=0, dim=0, new_value=sum([w.struct_info.shape[0] for w in weights])))
            assert decode_rewrite_success
            mod[decode_combine_func_name] = new_decode_tir
            new_decode_gvar = mod.get_global_var(decode_combine_func_name)
            relax.expr._update_struct_info(new_decode_gvar, mod[decode_combine_func_name].struct_info)
        else:
            new_decode_gvar = mod.get_global_var(decode_combine_func_name)

        if not any(gvar.name_hint == matmul_combine_func_name for gvar in mod.functions ):
            matmul_tir = mod[matmul_gvar]
            new_matmul_tir, matmul_rewrite_success = rewrite_tir(matmul_tir, ParamRewriteSpec(param_index=1, dim=0, new_value=sum([w.struct_info.shape[0] for w in weights])))
            assert matmul_rewrite_success
            mod[matmul_combine_func_name] = new_matmul_tir
            new_matmul_gvar = mod.get_global_var(matmul_combine_func_name)
            relax.expr._update_struct_info(new_matmul_gvar, mod[matmul_combine_func_name].struct_info)
        else:
            new_matmul_gvar = mod.get_global_var(matmul_combine_func_name)
            
        concat_wqkv_shape = list(weights[0].struct_info.shape)
        concat_wqkv_shape[0] = sum([w.struct_info.shape[0] for w in weights])
        concat_wqkv_sinfo = R.Tensor(concat_wqkv_shape, dtype="float16")
        concat_wqkv = R.call_tir(
            new_decode_gvar,
            (concat_wqkv_data, concat_wqkv_scale),
            out_sinfo=concat_wqkv_sinfo
        )
        concat_qkv_shape = list(matmul_outputs[0].struct_info.shape)
        concat_qkv_shape[-1] = sum([o.struct_info.shape[-1] for o in matmul_outputs])
        concat_qkv_sinfo = R.Tensor(concat_qkv_shape, dtype="float16")
        concat_qkv = R.call_tir(
            new_matmul_gvar,
            (input, concat_wqkv),
            out_sinfo=concat_qkv_sinfo
        )
        split_idx = [sum([o.struct_info.shape[-1] for o in matmul_outputs[:i]]) for i in range(1, len(matmul_outputs))]
        qkv_tuple = R.split(concat_qkv, split_idx, axis=-1)
        out_dict = {matmul_outputs[i]: R.TupleGetItem(qkv_tuple, i) for i in range(len(matmul_outputs))}
        
        return out_dict
    func = rewrite_bindings(ctx, rewriter, func)
    return func

def combine_qkv():
    @tvm.ir.transform.module_pass(opt_level=0, name="fuse_split_rotary_embedding")
    def ir_module_pass(mod: tvm.IRModule, _pass_context) -> tvm.IRModule:
        new_mod = {}
        for gvar, func in mod.functions.items():
            if isinstance(func, relax.Function):
                func = combine_parallel_matmul(mod, func, num_parallel=3)
                func = combine_parallel_matmul(mod, func, num_parallel=2)
            new_mod[gvar] = func
            
        for gvar, func in mod.functions.items():
            if isinstance(func, tir.PrimFunc) and gvar not in new_mod:
                new_mod[gvar] = func

        new_mod = tvm.IRModule(new_mod, mod.type_definitions, mod.attrs, mod.global_infos)
        return new_mod

    return ir_module_pass
