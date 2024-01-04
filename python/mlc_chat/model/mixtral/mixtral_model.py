"""
Implementation for Mistral architecture.
"""
import dataclasses

from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.topi.cuda.scan import inclusive_scan

from mlc_chat import op as op_ext
from mlc_chat.model.mistral.mistral_model import (
    MistralAttention,
    MistralConfig,
    MistralForCasualLM,
    MistralModel,
    RotaryEmbedding,
)
from mlc_chat.nn.expert import MixtralExperts
from mlc_chat.support import logging
from mlc_chat.support import tensor_parallel as tp

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MixtralConfig(MistralConfig):  # pylint: disable=too-many-instance-attributes
    """Configuration of the Mixtral model."""

    num_local_experts: int = 0
    num_experts_per_tok: int = 0


# pylint: disable=invalid-name,missing-docstring,too-many-locals,fixme

class MixtralMoE(nn.Module):
    """Mixture of experts"""

    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.gate = nn.Linear(
            in_features=config.hidden_size, out_features=config.num_local_experts, bias=False
        )
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_local_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size // config.tensor_parallel_shards
        self.e1_e3 = MixtralExperts(
            self.num_local_experts,
            self.num_experts_per_tok,
            in_features=config.hidden_size,
            out_features=2 * self.intermediate_size,
        )
        self.e2 = MixtralExperts(
            self.num_local_experts,
            self.num_experts_per_tok,
            in_features=self.intermediate_size,
            out_features=config.hidden_size,
        )
        self.dtype = "float32"

    # TODO: replace with cumsum nn op when it's ready
    def cumsum(self, data: Tensor, dim: int) -> Tensor:
        return op.tensor_expr_op(inclusive_scan, "cumsum", args=[data, dim, "int32"])

    def sum(self, x):
        # dlight cannot handle too small reduction axis extent
        # so we manually transform it into spatial op.
        if self.num_experts_per_tok == 2:

            def te_add(x):
                new_shape = (x.shape[0], x.shape[2])
                return te.compute(
                    new_shape,
                    lambda i, j: x[i, 0, j] + x[i, 1, j],
                    name="add",
                )

            return op.tensor_expr_op(te_add, "topk_mask", args=[x])
        return op.sum(x, axis=1)

    def forward(self, x: Tensor):
        assert x.ndim == 3
        input_shape = x.shape
        x = op.reshape(x, (input_shape[0] * input_shape[1], input_shape[2]))
        num_tokens = input_shape[0] * input_shape[1]

        # MoE data preparation
        gate: Tensor = self.gate(x)
        expert_weights, expert_indices = op_ext.topk(
            gate, self.num_experts_per_tok, self.num_local_experts, self.dtype, "int32"
        )
        expert_weights = op.softmax(expert_weights.astype("float32"), axis=-1).astype(self.dtype)
        if num_tokens == 1:
            # single batch decode
            expert_indices = op.reshape(expert_indices, (self.num_experts_per_tok,))
            concat_x1_x3 = self.e1_e3(x, expert_indices, single_batch_decode=True)
            x1, x3 = op.split(concat_x1_x3, indices_or_sections=2, axis=-1)
            linear_out = self.e2(op.silu(x1) * x3, expert_indices, single_batch_decode=True)
            unflattened = op.reshape(
                linear_out, (num_tokens, self.num_experts_per_tok, linear_out.shape[-1])
            )
        else:
            expert_mask = op_ext.topk_mask(
                expert_indices, self.num_experts_per_tok, self.num_local_experts
            )
            mask_T_flattened = op.reshape(
                op.permute_dims(expert_mask), (expert_mask.shape[0] * expert_mask.shape[1],)
            )
            cumsum_colwise_flattened = self.cumsum(mask_T_flattened, dim=0)
            flattened_indices = op_ext.get_indices(
                cumsum_colwise_flattened, expert_indices, self.num_experts_per_tok
            )
            indptr = op_ext.get_indptr(cumsum_colwise_flattened, self.num_local_experts)
            token_indices = op.divide(
                flattened_indices, Tensor.from_const(self.num_experts_per_tok)
            )
            gathered_x = op.take(x, token_indices, axis=0)

            # expert forward begin
            concat_x1_x3 = self.e1_e3(gathered_x, indptr)
            x1, x3 = op.split(concat_x1_x3, indices_or_sections=2, axis=-1)
            linear_out = self.e2(op.silu(x1) * x3, indptr)
            # expert forward end

            # MoE result post-processing
            unpermuted = op_ext.scatter_output(flattened_indices, linear_out, self.dtype)
            unflattened = op.reshape(
                unpermuted, (num_tokens, self.num_experts_per_tok, unpermuted.shape[1])
            )
        expert_weights = op.reshape(expert_weights, (num_tokens, self.num_experts_per_tok, 1))
        weighted_sum = self.sum(unflattened * expert_weights)
        weighted_sum = op.reshape(
            weighted_sum, (input_shape[0], input_shape[1], weighted_sum.shape[-1])
        )
        return weighted_sum


class MixtralDecoderLayer(nn.Module):
    """Mixtral decoder layer"""

    def __init__(self, config: MixtralConfig, rotary_embedding: RotaryEmbedding):
        rms_norm_eps = config.rms_norm_eps
        self.self_attn = MistralAttention(config, rotary_embedding)
        self.moe = MixtralMoE(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)

        def _set_tp():
            def _set(layer, hint):
                layer.weight.attrs["shard_strategy"] = hint

            hd = config.head_dim
            q = self.self_attn.num_q_heads * hd
            k = self.self_attn.num_kv_heads * hd
            v = self.self_attn.num_kv_heads * hd
            i = self.moe.intermediate_size
            _set(self.self_attn.qkv_proj, tp.ShardSingleDim("_shard_qkv", segs=[q, k, v], dim=0))
            _set(self.self_attn.o_proj, tp.ShardSingleDim("_shard_o", dim=1))
            _set(self.moe.e1_e3, tp.ShardSingleDim("_shard_mlp_up", segs=[i, i], dim=1))
            _set(self.moe.e2, tp.ShardSingleDim("_shard_mlp_down", dim=2))

        self.tensor_parallel_shards = config.tensor_parallel_shards
        _set_tp()

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
            if self.tensor_parallel_shards > 1:
                return op.ccl_allreduce(out + residual / self.tensor_parallel_shards, "sum")
            return out + residual

        out = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask,
            rolling_cache_len,
            kv_seq_len,
            cache_offset,
        )
        hidden_states = _apply_residual(out, residual=hidden_states)
        out = self.moe(self.post_attention_layernorm(hidden_states))
        hidden_states = _apply_residual(out, residual=hidden_states)
        return hidden_states


class MixtralModel(MistralModel):
    """Exact same as LlamaModel."""

    def __init__(self, config: MixtralConfig):
        super().__init__(config)
        rotary_embedding = RotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [MixtralDecoderLayer(config, rotary_embedding) for _ in range(config.num_hidden_layers)]
        )


class MixtralForCasualLM(MistralForCasualLM):
    """Same as LlamaForCausalLM, except for the use of sliding window attention."""

    def __init__(self, config: MixtralConfig):
        super().__init__(config)
        self.model = MixtralModel(config)
