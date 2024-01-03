"""Extern module for compiler."""
from .attention import attention
from .extern import configure, enable, get_store
from .gemm import faster_transformer_dequantize_gemm
from .moe import *
from .position_embedding import llama_rope
