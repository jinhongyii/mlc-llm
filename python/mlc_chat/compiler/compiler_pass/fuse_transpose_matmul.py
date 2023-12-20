"""A compiler pass that fuses transpose + matmul."""
import tvm
from tvm import IRModule, relax, te, tir
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.expr_functor import PyExprMutator, mutator


@tvm.transform.module_pass(opt_level=0, name="FuseTransposeMatmul")
class FuseTransposeMatmul:  # pylint: disable=too-few-public-methods
    """A compiler pass that fuses transpose + matmul."""

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """IRModule-level transformation"""
        mod = relax.transform.FuseOpsByPattern(
            [
                (
                    "transpose_matmul_fuse",
                    *_pattern(has_repeat, permute_x, permute_w),
                )
                for has_repeat in [True, False]
                for permute_x in [True, False]
                for permute_w in [True, False]
            ]
        )(mod)
        return mod

def _pattern(has_repeat=False, permute_x=False, permute_w=True):
    """Pattern for transpose + matmul."""
    # pylint: disable=invalid-name
    w = wildcard()
    x = wildcard()
    if has_repeat:
        w = is_op("relax.repeat")(w)
    if permute_x:
        x = is_op("relax.permute_dims")(x)
    wT = is_op("relax.permute_dims")(w)
    o = is_op("relax.matmul")(x, wT)
    # pylint: enable=invalid-name
    annotations = {}

    def _check(context: relax.transform.PatternCheckContext) -> bool:
        return True

    return o, annotations, _check


