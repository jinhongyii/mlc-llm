from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_chat import op as op_ext


class MixtralExperts(nn.Module):
    """Mixtral experts"""

    def __init__(self, num_local_experts, num_experts_per_tok, in_features, out_features):
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter((num_local_experts, out_features, in_features))
        self.dtype = "float32"

    def forward(self, x: Tensor, indptr: Tensor, single_batch_decode: bool = False):
        assert x.ndim == 2
        if single_batch_decode:
            # single-batch decode
            assert x.shape[1] == self.in_features
            assert indptr.ndim == 1
            if x.shape[0] == 1:
                return op_ext.gemv_e1_e3(
                    x,
                    self.weight,
                    indptr,
                    self.in_features,
                    self.out_features,
                    self.num_experts_per_tok,
                    self.num_local_experts,
                    self.dtype,
                )
            return op_ext.gemv_e2(
                x,
                self.weight,
                indptr,
                self.in_features,
                self.out_features,
                self.num_experts_per_tok,
                self.num_local_experts,
                self.dtype,
            )

        return op_ext.group_gemm(
            x,
            self.weight,
            indptr,
            self.in_features,
            self.out_features,
            self.num_local_experts,
            self.dtype,
        )
