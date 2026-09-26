"""PQ-HSA: IVF-PQ sparse/approximate attention prototype."""

from pq_hsa.attention.decode_adapter import DecodeAttentionOutput, IVFPQDecodeAttentionAdapter
from pq_hsa.attention.kv_cache import FetchedKV, KVCacheMemoryStats, PrefetchedKV, SparseKVCache
from pq_hsa.attention.module import IVFPQDecodeAttentionModule, ModuleAttentionOutput
from pq_hsa.attention.projected import (
    IVFPQProjectedAttentionModule,
    ProjectedAttentionOutput,
    build_projected_attention_from_layer,
)
from pq_hsa.attention.sparse_attention import (
    AttentionOutput,
    SparseAttentionConfig,
    IVFPQSparseAttention,
    dense_attention,
)
from pq_hsa.attention.multihead import (
    IVFPQMultiHeadAttention,
    MultiHeadAttentionOutput,
    MultiQueryAttentionOutput,
)
from pq_hsa.index.ivfpq import IVFPQConfig, IVFPQIndex, IVFPQMemoryStats, SearchResult
from pq_hsa.index.packing import pack_4bit_codes, unpack_4bit_codes
from pq_hsa.index.pq import ProductQuantizer, ProductQuantizerConfig
from pq_hsa.kernels import (
    KernelTopKResult,
    final_topk_merge_triton,
    is_triton_available,
    list_exp_sums_triton,
    score_packed_4bit_lut,
    score_packed_4bit_lut_batched_list_bias_triton,
    score_packed_4bit_lut_batched_triton,
    score_packed_4bit_lut_triton,
    topk_packed_4bit_lut,
)

__all__ = [
    "AttentionOutput",
    "DecodeAttentionOutput",
    "FetchedKV",
    "IVFPQConfig",
    "IVFPQDecodeAttentionAdapter",
    "IVFPQDecodeAttentionModule",
    "IVFPQIndex",
    "IVFPQMemoryStats",
    "IVFPQMultiHeadAttention",
    "IVFPQProjectedAttentionModule",
    "IVFPQSparseAttention",
    "KernelTopKResult",
    "KVCacheMemoryStats",
    "MultiHeadAttentionOutput",
    "MultiQueryAttentionOutput",
    "ModuleAttentionOutput",
    "PrefetchedKV",
    "ProductQuantizer",
    "ProductQuantizerConfig",
    "ProjectedAttentionOutput",
    "SearchResult",
    "SparseAttentionConfig",
    "SparseKVCache",
    "build_projected_attention_from_layer",
    "dense_attention",
    "final_topk_merge_triton",
    "is_triton_available",
    "list_exp_sums_triton",
    "pack_4bit_codes",
    "score_packed_4bit_lut",
    "score_packed_4bit_lut_batched_list_bias_triton",
    "score_packed_4bit_lut_batched_triton",
    "score_packed_4bit_lut_triton",
    "topk_packed_4bit_lut",
    "unpack_4bit_codes",
]
