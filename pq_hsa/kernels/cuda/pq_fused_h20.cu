// H20 (sm_90) fused PQ-HSA decode — CUDA, not Triton.
// Kernel 1: one global pass (pair-LUT → smem scores) + CTA radix-select.
// Kernel 1b: merge local top-k + fold row_max / ret_exp / list_mass.
// Kernel 2: hybrid exact attend (warp-cooperative dots).
//
// Compile-time shape: G=PQ_HSA_KG (default 4; instances for 4/5/8), M=8
// (4 packed bytes), L<=512, D<=128.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <cstdlib>

namespace {

// The GQA group is a compile-time constant of this translation unit.
// It is NOT changed for the default build: without -DPQ_HSA_KG the preprocessor
// emits exactly `constexpr int kG = 4;` as before, so the adopted G=4 binary is
// byte-identical.  Extra instances (G=5, G=8) are produced by compiling this
// same source again under a different extension name with -DPQ_HSA_KG=<G>.
#ifndef PQ_HSA_KG
#define PQ_HSA_KG 4
#endif
constexpr int kG = PQ_HSA_KG;
static_assert(kG >= 1 && kG <= 16, "PQ_HSA_KG out of range");
constexpr int kPairs = 4;
constexpr int kLMax = 512;
constexpr int kBlock = 256;
constexpr int kMergeMax = 1024;
constexpr int kDMax = 128;
// KG=16 instance.  The dynamic-smem scan tile scales with kG and would
// exceed the 227 KiB H20 limit at kG=16 with a 4096 tile; the G=16 build
// passes -DPQ_HSA_MAXTILE=2048.  Without the -D the emitted constexpr is
// unchanged (4096), so every existing instance keeps its codegen.
#ifndef PQ_HSA_MAXTILE
#define PQ_HSA_MAXTILE 4096
#endif
constexpr int kMaxTile = PQ_HSA_MAXTILE;
static_assert(kMaxTile % 256 == 0 && kMaxTile >= 256, "PQ_HSA_MAXTILE must be a multiple of 256");
constexpr float kLog2e = 1.4426950408889634f;

__device__ __forceinline__ uint16_t fp16_sort_key(float s) {
    __half h = __float2half(s);
    uint16_t bits = __half_as_ushort(h);
    if (bits & 0x8000u) {
        return static_cast<uint16_t>((~bits) & 0xFFFFu);
    }
    return static_cast<uint16_t>(bits ^ 0x8000u);
}

__device__ __forceinline__ void score4(
    uint32_t word,
    int lid,
    float kn,
    const __half* __restrict__ lut,
    const __half* __restrict__ lsc,
    float acc[kG]
) {
#pragma unroll
    for (int g = 0; g < kG; ++g) acc[g] = 0.f;
#pragma unroll
    for (int p = 0; p < kPairs; ++p) {
        int byte = static_cast<int>((word >> (8 * p)) & 0xFFu);
        const __half* row = lut + (p * 256 + byte) * kG;
#pragma unroll
        for (int g = 0; g < kG; ++g) {
            acc[g] += __half2float(row[g]);
        }
    }
    const __half* ls = lsc + lid * kG;
#pragma unroll
    for (int g = 0; g < kG; ++g) {
        acc[g] = (acc[g] + __half2float(ls[g])) * kn;
    }
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
    }
    return v;
}

__device__ __forceinline__ void hist_add(int* hist, int bin, bool valid) {
    unsigned mask = __match_any_sync(0xffffffffu, valid ? bin : -1);
    if (valid && (threadIdx.x & 31) == (__ffs(mask) - 1)) {
        atomicAdd(&hist[bin], __popc(mask));
    }
}

__device__ __forceinline__ void compact_write(
    bool pred, int* cnt, int cap, int base_slot,
    int32_t* idx_out, float* val_out, int tok, float s
) {
    unsigned m = __ballot_sync(0xffffffffu, pred);
    int lane = threadIdx.x & 31;
    int prefix = __popc(m & ((1u << lane) - 1));
    int n = __popc(m);
    int base = 0;
    if (lane == 0 && n) base = atomicAdd(cnt, n);
    base = __shfl_sync(0xffffffffu, base, 0);
    int slot = base + prefix;
    if (pred && slot < cap) {
        idx_out[base_slot + slot] = tok;
        val_out[base_slot + slot] = s;
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, m);
    }
    return v;
}

// List-sorted tokens: a warp of 32 consecutive ids is usually 1 list.
// Uniform warp → one warp-sum + one atomic; else match_any + shfl gather.
__device__ __forceinline__ void seg_mass_add(float* smass, int lid, float e, bool valid) {
    const int lane = threadIdx.x & 31;
    const int vlid = valid ? lid : -1;
    const int ref = __shfl_sync(0xffffffffu, vlid, 0);
    const bool uni = (ref >= 0) && __all_sync(0xffffffffu, !valid || vlid == ref);
    if (uni) {
        float sum = warp_sum(valid ? e : 0.f);
        if (lane == 0) atomicAdd(&smass[ref], sum);
        return;
    }
    const unsigned mask = __match_any_sync(0xffffffffu, vlid);
    if (!valid) return;
    float sum = 0.f;
    unsigned m = mask;
    while (m) {
        int src = __ffs(m) - 1;
        sum += __shfl_sync(mask, e, src);
        m &= ~(1u << src);
    }
    if (lane == (__ffs(mask) - 1)) atomicAdd(&smass[lid], sum);
}

__device__ __forceinline__ void load4_u2(
    const __half* __restrict__ base, int64_t row, int D, int dbase, bool on, float o[4]
) {
    if (!on) {
        o[0] = o[1] = o[2] = o[3] = 0.f;
        return;
    }
    const uint2 u = *reinterpret_cast<const uint2*>(base + row * D + dbase);
    const __half* hp = reinterpret_cast<const __half*>(&u);
    o[0] = __half2float(hp[0]);
    o[1] = __half2float(hp[1]);
    o[2] = __half2float(hp[2]);
    o[3] = __half2float(hp[3]);
}

__device__ __forceinline__ float block_reduce_max(float v, float* smem) {
    v = warp_max(v);
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : -INFINITY;
    if (wid == 0) v = warp_max(v);
    if (threadIdx.x == 0) smem[0] = v;
    __syncthreads();
    v = smem[0];
    __syncthreads();
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
    v = warp_sum(v);
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : 0.f;
    if (wid == 0) v = warp_sum(v);
    if (threadIdx.x == 0) smem[0] = v;
    __syncthreads();
    v = smem[0];
    __syncthreads();
    return v;
}

int choose_ctas_host(int N) {
    int C = (N + kMaxTile - 1) / kMaxTile;
    if (C < 1) C = 1;
    if (C > 64) C = 64;
    return C;
}

// ---------------------------------------------------------------------------
// Kernel 1: per-(head, cta) one global read + smem radix-select.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 2) pq_scan_select_kernel(
    const uint32_t* __restrict__ packed,
    const int32_t* __restrict__ list_ids,
    const __half* __restrict__ pair_lut,
    const __half* __restrict__ list_sc,
    const __half* __restrict__ token_scale,
    int32_t* __restrict__ cand_idx,
    float* __restrict__ cand_val,
    float* __restrict__ block_max,
    float* __restrict__ block_expsum,
    float* __restrict__ list_partial,
    __half* __restrict__ score_out,
    int64_t* __restrict__ clocks,  // 8 slots, CTA0/head0 only; nullable
    int N,
    int C,
    int K,
    int L,
    int packed_head_stride,
    int ids_head_stride,
    int scale_head_stride,
    int has_scale,
    int mass_mode
) {
    const int cta = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int tile = (N + C - 1) / C;
    const int begin = cta * tile;
    const int end = min(begin + tile, N);
    const int nloc = max(end - begin, 0);
    const int kloc = min(K, max(nloc, 1));
    const bool fit = (nloc <= kMaxTile);
    const int nstore = fit ? nloc : 0;

    extern __shared__ char dyn[];
    __half* slib = reinterpret_cast<__half*>(dyn);
    __half* slsc = slib + (kPairs * 256 * kG);
    __half* ssc = slsc + (kLMax * kG);
    uint16_t* slids = reinterpret_cast<uint16_t*>(ssc + (kMaxTile * kG));
    // slids ends at a 16-byte boundary (45056+8192=53248). Do NOT
    // uintptr_t-align smem pointers — that drops the shared address space.
    int* hist = reinterpret_cast<int*>(slids + kMaxTile);
    int* hist_lo = hist + (kG * 256);
    float* smass = reinterpret_cast<float*>(hist_lo + (kG * 256));
    float* red = smass + (kG * kLMax);
    int* bstar = reinterpret_cast<int*>(red + 32);
    int* cabove = bstar + kG;
    int* lostar = cabove + kG;
    int* neq = lostar + kG;
    int* tstar = neq + kG;
    int* cnt_gt = tstar + kG;
    int* cnt_eq = cnt_gt + kG;
    float* bmax_s = reinterpret_cast<float*>(cnt_eq + kG);
    float* bexp_s = bmax_s + kG;

    const bool prof = (clocks != nullptr && cta == 0 && h == 0 && tid == 0);
    int64_t c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0, c7 = 0;
    if (prof) c0 = clock64();

    const __half* hlut = pair_lut + h * (kPairs * 256 * kG);
    const __half* hlsc = list_sc + h * (L * kG);
    for (int i = tid; i < kPairs * 256 * kG; i += kBlock) slib[i] = hlut[i];
    for (int i = tid; i < L * kG; i += kBlock) slsc[i] = hlsc[i];
    for (int i = tid; i < kG * 256; i += kBlock) {
        hist[i] = 0;
        hist_lo[i] = 0;
    }
    for (int i = tid; i < kG * kLMax; i += kBlock) smass[i] = 0.f;
    if (tid < kG) {
        cnt_gt[tid] = 0;
        cnt_eq[tid] = 0;
        bmax_s[tid] = -INFINITY;
        bexp_s[tid] = 0.f;
        neq[tid] = 0;
    }
    __syncthreads();
    if (prof) c1 = clock64();

    const uint32_t* hpk = packed + h * packed_head_stride;
    const int32_t* hid = list_ids + h * ids_head_stride;
    const __half* hsc = has_scale ? (token_scale + h * scale_head_stride) : nullptr;

    // ---- Pass A: score tile into smem, hi-hist, thread max ----
    float tmax[kG];
#pragma unroll
    for (int g = 0; g < kG; ++g) tmax[g] = -INFINITY;

    const int niter_a = (nloc + kBlock - 1) / kBlock;
    for (int it = 0; it < niter_a; ++it) {
        const int t = begin + tid + it * kBlock;
        const bool valid = t < end;
        uint32_t word = valid ? __ldg(hpk + t) : 0u;
        int lid = valid ? static_cast<int>(__ldg(hid + t)) : 0;
        if (lid < 0) lid = 0;
        if (lid >= L) lid = 0;
        float kn = (valid && has_scale) ? __half2float(__ldg(hsc + t)) : 1.f;
        float acc[kG];
        score4(word, lid, kn, slib, slsc, acc);
        const int local = t - begin;
        if (valid && fit && local < kMaxTile) {
            slids[local] = static_cast<uint16_t>(lid);
#pragma unroll
            for (int g = 0; g < kG; ++g) {
                ssc[local * kG + g] = __float2half(acc[g]);
            }
        }
        if (valid && score_out != nullptr) {
#pragma unroll
            for (int g = 0; g < kG; ++g) {
                score_out[(static_cast<int64_t>(h) * kG + g) * N + t] = __float2half(acc[g]);
            }
        }
#pragma unroll
        for (int g = 0; g < kG; ++g) {
            if (valid) tmax[g] = fmaxf(tmax[g], acc[g]);
            uint16_t key = fp16_sort_key(valid ? acc[g] : -1e30f);
            hist_add(&hist[g * 256], static_cast<int>(key >> 8), valid);
        }
    }
#pragma unroll
    for (int g = 0; g < kG; ++g) {
        float m = block_reduce_max(tmax[g], red);
        if (tid == 0) bmax_s[g] = m;
    }
    __syncthreads();

    if (tid < kG) {
        int need = kloc;
        int above = 0;
        int bs = 0;
        for (int b = 255; b >= 0; --b) {
            int c = hist[tid * 256 + b];
            if (above + c >= need) {
                bs = b;
                break;
            }
            above += c;
        }
        bstar[tid] = bs;
        cabove[tid] = above;
    }
    __syncthreads();
    if (prof) c2 = clock64();

    // ---- Pass B: one exp2 / token: texp + segmented mass + lo-hist(b*) ----
    float texp[kG];
#pragma unroll
    for (int g = 0; g < kG; ++g) texp[g] = 0.f;
    const int niter_b = (nloc + kBlock - 1) / kBlock;
    for (int it = 0; it < niter_b; ++it) {
        const int t = begin + tid + it * kBlock;
        const bool valid = t < end;
        int lid = 0;
        float acc[kG];
#pragma unroll
        for (int g = 0; g < kG; ++g) acc[g] = 0.f;
        if (valid && fit) {
            const int local = t - begin;
            lid = static_cast<int>(slids[local]);
#pragma unroll
            for (int g = 0; g < kG; ++g) acc[g] = __half2float(ssc[local * kG + g]);
        } else if (valid) {
            uint32_t word = __ldg(hpk + t);
            lid = static_cast<int>(__ldg(hid + t));
            if (lid < 0) lid = 0;
            if (lid >= L) lid = 0;
            float kn = has_scale ? __half2float(__ldg(hsc + t)) : 1.f;
            score4(word, lid, kn, slib, slsc, acc);
        }
#pragma unroll
        for (int g = 0; g < kG; ++g) {
            float e = 0.f;
            if (valid) {
                e = exp2f((acc[g] - bmax_s[g]) * kLog2e);
                texp[g] += e;
            }
            if (mass_mode == 0) {
                seg_mass_add(&smass[g * kLMax], lid, e, valid);
            }
            uint16_t key = fp16_sort_key(valid ? acc[g] : -1e30f);
            bool in_hi = valid && ((int)(key >> 8) == bstar[g]);
            hist_add(&hist_lo[g * 256], static_cast<int>(key & 0xFF), in_hi);
        }
    }
#pragma unroll
    for (int g = 0; g < kG; ++g) {
        float s = block_reduce_sum(texp[g], red);
        if (tid == 0) bexp_s[g] = s;
    }
    if (tid < kG) {
        int need = max(kloc - cabove[tid], 0);
        int above = 0;
        int ls = 0;
        int n_eq = 0;
        for (int b = 255; b >= 0; --b) {
            int c = hist_lo[tid * 256 + b];
            if (above + c >= need) {
                ls = b;
                n_eq = need - above;
                break;
            }
            above += c;
        }
        lostar[tid] = ls;
        tstar[tid] = (bstar[tid] << 8) + ls;
        neq[tid] = max(n_eq, 0);
    }
    __syncthreads();
    if (prof) c3 = clock64();

    // ---- Pass C: compact collect (smem if fit, else rescore) ----
    int32_t* gidx = cand_idx + (((h * C + cta) * kG) * K);
    float* gval = cand_val + (((h * C + cta) * kG) * K);

    const int niter_c = (nloc + kBlock - 1) / kBlock;
    for (int it = 0; it < niter_c; ++it) {
        const int t = begin + tid + it * kBlock;
        const bool valid = t < end;
        float acc[kG];
#pragma unroll
        for (int g = 0; g < kG; ++g) acc[g] = -1e30f;
        if (valid && fit) {
            const int local = t - begin;
#pragma unroll
            for (int g = 0; g < kG; ++g) acc[g] = __half2float(ssc[local * kG + g]);
        } else if (valid) {
            uint32_t word = __ldg(hpk + t);
            int lid = static_cast<int>(__ldg(hid + t));
            if (lid < 0) lid = 0;
            if (lid >= L) lid = 0;
            float kn = has_scale ? __half2float(__ldg(hsc + t)) : 1.f;
            score4(word, lid, kn, slib, slsc, acc);
        }
#pragma unroll
        for (int g = 0; g < kG; ++g) {
            float s = acc[g];
            int key = static_cast<int>(fp16_sort_key(s));
            int ts = tstar[g];
            compact_write(valid && key > ts, &cnt_gt[g], kloc, g * K, gidx, gval, t, s);
            if (valid && key == ts) {
                int slot = atomicAdd(&cnt_eq[g], 1);
                if (slot < neq[g]) {
                    int pos = kloc - 1 - slot;
                    gidx[g * K + pos] = t;
                    gval[g * K + pos] = s;
                }
            }
        }
    }
    __syncthreads();

    for (int i = tid; i < kG * K; i += kBlock) {
        int j = i % K;
        if (j >= kloc) {
            gidx[i] = 0;
            gval[i] = -INFINITY;
        }
    }
    if (tid < kG) {
        block_max[(h * C + cta) * kG + tid] = bmax_s[tid];
        block_expsum[(h * C + cta) * kG + tid] = bexp_s[tid];
    }
    if (mass_mode == 0) {
        float* lp = list_partial + ((h * C + cta) * kG) * kLMax;
        for (int i = tid; i < kG * kLMax; i += kBlock) {
            lp[i] = (i % kLMax < L) ? smass[i] : 0.f;
        }
    }
    if (prof) {
        c4 = clock64();
        clocks[0] = c1 - c0; // lut+zero
        clocks[1] = c2 - c1; // passA score+hihist+max
        clocks[2] = c3 - c2; // passB lohist+mass
        clocks[3] = c4 - c3; // collect+write
        clocks[4] = c4 - c0; // total
    }
}

// ---------------------------------------------------------------------------
// Merge C local top-k lists from global memory (hist in smem only).
// ---------------------------------------------------------------------------
__global__ void pq_merge_topk_kernel(
    const int32_t* __restrict__ cand_idx,
    const float* __restrict__ cand_val,
    int32_t* __restrict__ out_idx,
    float* __restrict__ out_val,
    const float* __restrict__ block_max,
    const float* __restrict__ block_expsum,
    const float* __restrict__ list_partial,
    float* __restrict__ row_max,
    float* __restrict__ ret_exp,
    float* __restrict__ list_mass,
    int C,
    int K,
    int L,
    int do_mass
) {
    const int h = blockIdx.x;
    const int g = blockIdx.y;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int n = C * K;

    __shared__ int hist[512];
    __shared__ int bstar, cabove, lostar, neq, tstar, cgt, ceq;
    __shared__ float rmax_s;

    for (int i = tid; i < 256; i += nthreads) hist[i] = 0;
    __syncthreads();
    const int niter = (n + nthreads - 1) / nthreads;
    for (int it = 0; it < niter; ++it) {
        const int i = tid + it * nthreads;
        const bool valid = i < n;
        float v = valid ? cand_val[(((h * C) + (i / K)) * kG + g) * K + (i % K)] : -1e30f;
        uint16_t key = fp16_sort_key(v);
        hist_add(&hist[0], static_cast<int>(key >> 8), valid);
    }
    __syncthreads();
    if (tid == 0) {
        int above = 0, bs = 0;
        for (int b = 255; b >= 0; --b) {
            if (above + hist[b] >= K) {
                bs = b;
                break;
            }
            above += hist[b];
        }
        bstar = bs;
        cabove = above;
    }
    __syncthreads();
    for (int i = tid; i < 256; i += nthreads) hist[256 + i] = 0;
    __syncthreads();
    for (int it = 0; it < niter; ++it) {
        const int i = tid + it * nthreads;
        const bool valid = i < n;
        float v = valid ? cand_val[(((h * C) + (i / K)) * kG + g) * K + (i % K)] : -1e30f;
        uint16_t key = fp16_sort_key(v);
        bool in = valid && ((int)(key >> 8) == bstar);
        hist_add(&hist[256], static_cast<int>(key & 0xFF), in);
    }
    __syncthreads();
    if (tid == 0) {
        int need = max(K - cabove, 0);
        int above = 0, ls = 0, n_eq = 0;
        for (int b = 255; b >= 0; --b) {
            if (above + hist[256 + b] >= need) {
                ls = b;
                n_eq = need - above;
                break;
            }
            above += hist[256 + b];
        }
        lostar = ls;
        neq = max(n_eq, 0);
        tstar = (bstar << 8) + ls;
        cgt = 0;
        ceq = 0;
    }
    __syncthreads();

    int32_t* oidx = out_idx + (h * kG + g) * K;
    float* oval = out_val + (h * kG + g) * K;
    for (int it = 0; it < niter; ++it) {
        const int i = tid + it * nthreads;
        if (i >= n) continue;
        int cta = i / K;
        int j = i % K;
        float v = cand_val[(((h * C) + cta) * kG + g) * K + j];
        int key = static_cast<int>(fp16_sort_key(v));
        if (key > tstar) {
            int slot = atomicAdd(&cgt, 1);
            if (slot < K) {
                oidx[slot] = cand_idx[(((h * C) + cta) * kG + g) * K + j];
                oval[slot] = v;
            }
        } else if (key == tstar) {
            int slot = atomicAdd(&ceq, 1);
            if (slot < neq) {
                int pos = K - 1 - slot;
                oidx[pos] = cand_idx[(((h * C) + cta) * kG + g) * K + j];
                oval[pos] = v;
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        float rmax = -INFINITY;
        for (int c = 0; c < C; ++c) {
            rmax = fmaxf(rmax, block_max[(h * C + c) * kG + g]);
        }
        float rexp = 0.f;
        for (int c = 0; c < C; ++c) {
            float bm = block_max[(h * C + c) * kG + g];
            float scale = (bm == -INFINITY) ? 0.f : exp2f((bm - rmax) * kLog2e);
            rexp += block_expsum[(h * C + c) * kG + g] * scale;
        }
        row_max[h * kG + g] = rmax;
        ret_exp[h * kG + g] = rexp;
        rmax_s = rmax;
    }
    __syncthreads();
    if (do_mass) {
        float rmax = rmax_s;
        float* lm = list_mass + (h * kG + g) * L;
        for (int ell = tid; ell < L; ell += nthreads) {
            float s = 0.f;
            for (int c = 0; c < C; ++c) {
                float bm = block_max[(h * C + c) * kG + g];
                float scale = (bm == -INFINITY) ? 0.f : exp2f((bm - rmax) * kLog2e);
                float m = list_partial[(((h * C + c) * kG + g) * kLMax) + ell];
                s += m * scale;
            }
            lm[ell] = s;
        }
    } else {
        for (int ell = tid; ell < L; ell += nthreads) {
            list_mass[(h * kG + g) * L + ell] = 0.f;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: hybrid exact attend. 128 threads = 4 warps × 32 (4 dims each).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void merge_warp_states(
    float w_max,
    float w_lse,
    float acc[4],
    int dbase,
    float* sm_max,
    float* sm_lse,
    float* sm_acc,
    float* out_max,
    float* out_lse,
    float out_acc[4]
) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) {
        sm_max[wid] = w_max;
        sm_lse[wid] = w_lse;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sm_acc[wid * kDMax + dbase + i] = acc[i];
    }
    __syncthreads();
    float m = sm_max[0];
#pragma unroll
    for (int w = 1; w < 4; ++w) m = fmaxf(m, sm_max[w]);
    float lse = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) out_acc[i] = 0.f;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        float so = (sm_max[w] == -INFINITY) ? 0.f : exp2f((sm_max[w] - m) * kLog2e);
        lse += sm_lse[w] * so;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            out_acc[i] += sm_acc[w * kDMax + dbase + i] * so;
        }
    }
    *out_max = m;
    *out_lse = lse;
    __syncthreads();
}

__global__ void __launch_bounds__(128, 4) pq_exact_attend_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ full_k,
    const __half* __restrict__ full_v,
    const float* __restrict__ mask,
    const int32_t* __restrict__ topk_idx,
    const float* __restrict__ topk_val,
    const int64_t* __restrict__ ret_global,
    const __half* __restrict__ sb_k,
    const __half* __restrict__ sb_v,
    const int32_t* __restrict__ list_ids,
    const float* __restrict__ list_mass,
    const __half* __restrict__ centroids,
    const float* __restrict__ row_max_in,
    const float* __restrict__ ret_exp_in,
    __half* __restrict__ out,
    int D, int N, int K, int F, int L, int CAP,
    float scale,
    int mode
) {
    const int h = blockIdx.x;
    const int g = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int dbase = lane * 4;
    const bool on = (dbase < D);
    const bool do_v = (mode != 1);
    const bool do_exact = (mode != 2);
    const bool do_full = (mode != 3);

    float qv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int d = dbase + i;
        qv[i] = (on && d < D) ? __half2float(q[(h * kG + g) * D + d]) : 0.f;
    }

    extern __shared__ char dyn2[];
    float* sm_max = reinterpret_cast<float*>(dyn2);
    float* sm_lse = sm_max + 4;
    float* sm_acc = sm_lse + 4;

    auto load_kv = [&](const __half* base, int64_t row, float o[4]) {
        load4_u2(base, row, D, dbase, on, o);
    };

    auto online = [&](float logit, const float vv[4],
                      float& mx, float& lse, float acc[4]) {
        float nm = fmaxf(mx, logit);
        float so = (mx == -INFINITY) ? 0.f : exp2f((mx - nm) * kLog2e);
        float e = (logit == -INFINITY) ? 0.f : exp2f((logit - nm) * kLog2e);
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] = acc[i] * so + e * vv[i];
        lse = lse * so + e;
        mx = nm;
    };

    float full_max = -INFINITY, full_lse = 0.f, full_acc[4] = {0, 0, 0, 0};
    if (do_full) for (int f = wid; f < F; f += 4) {
        float kv[4], vv[4] = {0, 0, 0, 0};
        load_kv(full_k, static_cast<int64_t>(h) * F + f, kv);
        if (do_v) load_kv(full_v, static_cast<int64_t>(h) * F + f, vv);
        float part = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) part += qv[i] * kv[i];
        float logit = warp_sum(part) * scale + mask[(h * kG + g) * F + f];
        online(logit, vv, full_max, full_lse, full_acc);
    }
    float fmax_m, flse_m, facc_m[4];
    merge_warp_states(full_max, full_lse, full_acc, dbase, sm_max, sm_lse, sm_acc,
                      &fmax_m, &flse_m, facc_m);

    float exact_max = -INFINITY, exact_lse = 0.f, exact_acc[4] = {0, 0, 0, 0};
    float old_max = -INFINITY, old_lse = 0.f;
    const int32_t* tidx = topk_idx + (h * kG + g) * K;
    const float* tval = topk_val + (h * kG + g) * K;
    const int64_t* rg = ret_global + h * N;
    const int32_t* lids = list_ids + h * N;

    if (do_exact) for (int t = wid; t < K; t += 4) {
        int local = tidx[t];
        if (local < 0) local = 0;
        if (local >= N) local = 0;
        float approx = tval[t];
        int64_t glob = rg[local];
        int64_t flat = glob + static_cast<int64_t>(h) * CAP;
        float kv[4], vv[4] = {0, 0, 0, 0};
        load_kv(sb_k, flat, kv);
        if (do_v) load_kv(sb_v, flat, vv);
        float part = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) part += qv[i] * kv[i];
        float elogit = warp_sum(part) * scale;
        online(elogit, vv, exact_max, exact_lse, exact_acc);

        float om = fmaxf(old_max, approx);
        float oso = (old_max == -INFINITY) ? 0.f : exp2f((old_max - om) * kLog2e);
        old_lse = old_lse * oso + exp2f((approx - om) * kLog2e);
        old_max = om;
    }
    float emax_m, else_m, eacc_m[4];
    merge_warp_states(exact_max, exact_lse, exact_acc, dbase, sm_max, sm_lse, sm_acc,
                      &emax_m, &else_m, eacc_m);

    // merge old_approx (scalar) across warps
    if (lane == 0) {
        sm_max[wid] = old_max;
        sm_lse[wid] = old_lse;
    }
    __syncthreads();
    float omax_m = sm_max[0];
#pragma unroll
    for (int w = 1; w < 4; ++w) omax_m = fmaxf(omax_m, sm_max[w]);
    float olse_m = 0.f;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        float so = (sm_max[w] == -INFINITY) ? 0.f : exp2f((sm_max[w] - omax_m) * kLog2e);
        olse_m += sm_lse[w] * so;
    }
    __syncthreads();

    float ret_max = row_max_in[h * kG + g];
    float ret_sum = ret_exp_in[h * kG + g];
    float row_max = fmaxf(fmaxf(fmax_m, ret_max), fmaxf(emax_m, omax_m));
    float full_sum = flse_m * ((fmax_m == -INFINITY) ? 0.f : exp2f((fmax_m - row_max) * kLog2e));
    float exact_sum = else_m * ((emax_m == -INFINITY) ? 0.f : exp2f((emax_m - row_max) * kLog2e));
    float old_sum = olse_m * ((omax_m == -INFINITY) ? 0.f : exp2f((omax_m - row_max) * kLog2e));
    float ret_adj = ret_sum * exp2f((ret_max - row_max) * kLog2e);
    float denom = full_sum + ret_adj - old_sum + exact_sum;
    if (denom < 1e-16f) denom = 1e-16f;

    float full_scale = (fmax_m == -INFINITY) ? 0.f : exp2f((fmax_m - row_max) * kLog2e) / denom;
    float exact_scale = (emax_m == -INFINITY) ? 0.f : exp2f((emax_m - row_max) * kLog2e) / denom;

    float bg[4] = {0, 0, 0, 0};
    const float* mass = list_mass + (h * kG + g) * L;
    const __half* cents = centroids + static_cast<int64_t>(h) * L * D;
    float scale_bg = exp2f((ret_max - row_max) * kLog2e);
    for (int ell = 0; ell < L; ++ell) {
        float m = mass[ell] * scale_bg;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int d = dbase + i;
            float c = (on && d < D) ? __half2float(cents[ell * D + d]) : 0.f;
            bg[i] += m * c;
        }
    }
    for (int t = 0; t < K; ++t) {
        int local = tidx[t];
        if (local < 0) local = 0;
        if (local >= N) local = 0;
        float approx = tval[t];
        int lid = lids[local];
        if (lid < 0) lid = 0;
        if (lid >= L) lid = 0;
        float old_m = exp2f((approx - row_max) * kLog2e);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int d = dbase + i;
            float c = (on && d < D) ? __half2float(cents[lid * D + d]) : 0.f;
            bg[i] -= old_m * c;
        }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int d = dbase + i;
        if (on && d < D) {
            float o = facc_m[i] * full_scale + eacc_m[i] * exact_scale + bg[i] / denom;
            out[(h * kG + g) * D + d] = __float2half(o);
        }
    }
}

// ---------------------------------------------------------------------------
// Paged-KV variant of pq_exact_attend_kernel (PQ_HSA_PAGED_ATTEND=1,
// opt-in). Purely additive -- pq_exact_attend_kernel above is byte-for-byte
// untouched, still reachable via pq_exact_attend()/cuda_attend_only() with
// the flag off.
//
// The ONLY thing that changes is where the exact-gather K/V rows come from:
// instead of a flat [H*CAP, D] `sb_k`/`sb_v` duplicate addressed by
// `retrieval_global[local] + h*CAP` (the sidecar's own resident copy of raw
// K/V, ~81-89% of sidecar bytes at 128K/272K), this
// kernel reads directly out of vLLM's own paged KV cache
// (`[2, num_blocks, block_size, num_kv_heads, head_dim]`, contiguous) via the
// request's `block_table` row, treating `retrieval_global[local]` as an
// ABSOLUTE token position in [0, seq_len) -- exactly the coordinate space
// vLLM's own block_table indexes (see benchmarks/vllm_backend/paged_kv_fa.py
// gather_flashattn_kv_at_indices, same convention, same math). The sink+local
// full-region read (`full_k`/`full_v`) and every other input/output/reduction
// is IDENTICAL to pq_exact_attend_kernel -- same value set, same warp
// reduction order, same fp32 accumulation -- so with a paged KV cache whose
// bytes equal the flat sb_k/sb_v duplicate at the same positions, the two
// kernels are expected to produce bit-identical output. Only the S<=1
// (non-split-K) path is implemented in this kernel; the split-K path is
// pq_attend_partial_nw_paged_kernel (see pq_exact_attend_paged_cuda below).
// ---------------------------------------------------------------------------
__device__ __forceinline__ int64_t pq_paged_slot(
    const int32_t* __restrict__ block_table, int64_t pos, int block_size
) {
    int64_t p = (pos < 0) ? 0 : pos;
    int64_t blk_idx = p / block_size;
    int64_t within = p - blk_idx * block_size;
    int64_t blk = static_cast<int64_t>(block_table[blk_idx]);
    return blk * static_cast<int64_t>(block_size) + within;
}

// (opt-in, vLLM >= 0.10 page layout): row index for load4_u2(base, row, D).
//   kv_layout 0: vLLM <= 0.8.5 5-D page [2, num_blocks, block_size, kv_heads, D];
//                 base = K or V plane, row = slot*kv_heads + h  (previous math, verbatim).
//   kv_layout 1: vLLM >= 0.10 4-D page [num_blocks, kv_heads, block_size, 2*D], K|V
//                 concatenated per slot; base_k = page ptr, base_v = page ptr + D,
//                 row = 2*((blk*kv_heads + h)*block_size + within) so row*D is the K half
//                 of that (blk, h, within) slot (uint2 alignment preserved: 2*D*2 B rows).
//   For kv_layout 1 the page is read IN PLACE through its logical strides (sB, sH, sN,
//   given in units of D elements), so any physical KVCacheLayout vLLM resolves (LBNHC,
//   LBHNC, BLNHC, ...) works without a .contiguous() copy; C (=2D, K|V) must be innermost.
__device__ __forceinline__ int64_t pq_paged_row(
    const int32_t* __restrict__ block_table, int64_t pos, int block_size,
    int num_kv_heads, int h, int kv_layout,
    int64_t sB, int64_t sH, int64_t sN
) {
    if (kv_layout == 0) {
        int64_t slot = pq_paged_slot(block_table, pos, block_size);
        return slot * static_cast<int64_t>(num_kv_heads) + h;
    }
    int64_t p = (pos < 0) ? 0 : pos;
    int64_t blk_idx = p / block_size;
    int64_t within = p - blk_idx * block_size;
    int64_t blk = static_cast<int64_t>(block_table[blk_idx]);
    return blk * sB + static_cast<int64_t>(h) * sH + within * sN;
}

__global__ void __launch_bounds__(128, 4) pq_exact_attend_paged_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ full_k,
    const __half* __restrict__ full_v,
    const float* __restrict__ mask,
    const int32_t* __restrict__ topk_idx,
    const float* __restrict__ topk_val,
    const int64_t* __restrict__ ret_global,   // ABSOLUTE token positions, not local-buf idx
    const __half* __restrict__ kv_k_base,     // vLLM kv_cache[0, ...], flat [num_blocks*block_size, num_kv_heads, D]
    const __half* __restrict__ kv_v_base,     // vLLM kv_cache[1, ...], same flat layout
    const int32_t* __restrict__ block_table,  // this request's row, [max_blocks]
    const int32_t* __restrict__ list_ids,
    const float* __restrict__ list_mass,
    const __half* __restrict__ centroids,
    const float* __restrict__ row_max_in,
    const float* __restrict__ ret_exp_in,
    __half* __restrict__ out,
    int D, int N, int K, int F, int L,
    int block_size, int num_kv_heads,
    float scale,
    int mode,
    int kv_layout, int64_t kv_sB, int64_t kv_sH, int64_t kv_sN
) {
    const int h = blockIdx.x;
    const int g = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int dbase = lane * 4;
    const bool on = (dbase < D);
    const bool do_v = (mode != 1);
    const bool do_exact = (mode != 2);
    const bool do_full = (mode != 3);

    float qv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int d = dbase + i;
        qv[i] = (on && d < D) ? __half2float(q[(h * kG + g) * D + d]) : 0.f;
    }

    extern __shared__ char dyn2[];
    float* sm_max = reinterpret_cast<float*>(dyn2);
    float* sm_lse = sm_max + 4;
    float* sm_acc = sm_lse + 4;

    auto load_kv = [&](const __half* base, int64_t row, float o[4]) {
        load4_u2(base, row, D, dbase, on, o);
    };

    auto load_kv_paged = [&](const __half* base, int64_t pos, float o[4]) {
        int64_t row = pq_paged_row(block_table, pos, block_size, num_kv_heads, h, kv_layout, kv_sB, kv_sH, kv_sN);
        load4_u2(base, row, D, dbase, on, o);
    };

    auto online = [&](float logit, const float vv[4],
                      float& mx, float& lse, float acc[4]) {
        float nm = fmaxf(mx, logit);
        float so = (mx == -INFINITY) ? 0.f : exp2f((mx - nm) * kLog2e);
        float e = (logit == -INFINITY) ? 0.f : exp2f((logit - nm) * kLog2e);
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] = acc[i] * so + e * vv[i];
        lse = lse * so + e;
        mx = nm;
    };

    float full_max = -INFINITY, full_lse = 0.f, full_acc[4] = {0, 0, 0, 0};
    if (do_full) for (int f = wid; f < F; f += 4) {
        float kv[4], vv[4] = {0, 0, 0, 0};
        load_kv(full_k, static_cast<int64_t>(h) * F + f, kv);
        if (do_v) load_kv(full_v, static_cast<int64_t>(h) * F + f, vv);
        float part = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) part += qv[i] * kv[i];
        float logit = warp_sum(part) * scale + mask[(h * kG + g) * F + f];
        online(logit, vv, full_max, full_lse, full_acc);
    }
    float fmax_m, flse_m, facc_m[4];
    merge_warp_states(full_max, full_lse, full_acc, dbase, sm_max, sm_lse, sm_acc,
                      &fmax_m, &flse_m, facc_m);

    float exact_max = -INFINITY, exact_lse = 0.f, exact_acc[4] = {0, 0, 0, 0};
    float old_max = -INFINITY, old_lse = 0.f;
    const int32_t* tidx = topk_idx + (h * kG + g) * K;
    const float* tval = topk_val + (h * kG + g) * K;
    const int64_t* rg = ret_global + h * N;
    const int32_t* lids = list_ids + h * N;

    if (do_exact) for (int t = wid; t < K; t += 4) {
        int local = tidx[t];
        if (local < 0) local = 0;
        if (local >= N) local = 0;
        float approx = tval[t];
        int64_t pos = rg[local];
        float kv[4], vv[4] = {0, 0, 0, 0};
        load_kv_paged(kv_k_base, pos, kv);
        if (do_v) load_kv_paged(kv_v_base, pos, vv);
        float part = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) part += qv[i] * kv[i];
        float elogit = warp_sum(part) * scale;
        online(elogit, vv, exact_max, exact_lse, exact_acc);

        float om = fmaxf(old_max, approx);
        float oso = (old_max == -INFINITY) ? 0.f : exp2f((old_max - om) * kLog2e);
        old_lse = old_lse * oso + exp2f((approx - om) * kLog2e);
        old_max = om;
    }
    float emax_m, else_m, eacc_m[4];
    merge_warp_states(exact_max, exact_lse, exact_acc, dbase, sm_max, sm_lse, sm_acc,
                      &emax_m, &else_m, eacc_m);

    // merge old_approx (scalar) across warps
    if (lane == 0) {
        sm_max[wid] = old_max;
        sm_lse[wid] = old_lse;
    }
    __syncthreads();
    float omax_m = sm_max[0];
#pragma unroll
    for (int w = 1; w < 4; ++w) omax_m = fmaxf(omax_m, sm_max[w]);
    float olse_m = 0.f;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        float so = (sm_max[w] == -INFINITY) ? 0.f : exp2f((sm_max[w] - omax_m) * kLog2e);
        olse_m += sm_lse[w] * so;
    }
    __syncthreads();

    float ret_max = row_max_in[h * kG + g];
    float ret_sum = ret_exp_in[h * kG + g];
    float row_max = fmaxf(fmaxf(fmax_m, ret_max), fmaxf(emax_m, omax_m));
    float full_sum = flse_m * ((fmax_m == -INFINITY) ? 0.f : exp2f((fmax_m - row_max) * kLog2e));
    float exact_sum = else_m * ((emax_m == -INFINITY) ? 0.f : exp2f((emax_m - row_max) * kLog2e));
    float old_sum = olse_m * ((omax_m == -INFINITY) ? 0.f : exp2f((omax_m - row_max) * kLog2e));
    float ret_adj = ret_sum * exp2f((ret_max - row_max) * kLog2e);
    float denom = full_sum + ret_adj - old_sum + exact_sum;
    if (denom < 1e-16f) denom = 1e-16f;

    float full_scale = (fmax_m == -INFINITY) ? 0.f : exp2f((fmax_m - row_max) * kLog2e) / denom;
    float exact_scale = (emax_m == -INFINITY) ? 0.f : exp2f((emax_m - row_max) * kLog2e) / denom;

    float bg[4] = {0, 0, 0, 0};
    const float* mass = list_mass + (h * kG + g) * L;
    const __half* cents = centroids + static_cast<int64_t>(h) * L * D;
    float scale_bg = exp2f((ret_max - row_max) * kLog2e);
    for (int ell = 0; ell < L; ++ell) {
        float m = mass[ell] * scale_bg;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int d = dbase + i;
            float c = (on && d < D) ? __half2float(cents[ell * D + d]) : 0.f;
            bg[i] += m * c;
        }
    }
    for (int t = 0; t < K; ++t) {
        int local = tidx[t];
        if (local < 0) local = 0;
        if (local >= N) local = 0;
        float approx = tval[t];
        int lid = lids[local];
        if (lid < 0) lid = 0;
        if (lid >= L) lid = 0;
        float old_m = exp2f((approx - row_max) * kLog2e);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int d = dbase + i;
            float c = (on && d < D) ? __half2float(cents[lid * D + d]) : 0.f;
            bg[i] -= old_m * c;
        }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int d = dbase + i;
        if (on && d < D) {
            float o = facc_m[i] * full_scale + eacc_m[i] * exact_scale + bg[i] / denom;
            out[(h * kG + g) * D + d] = __float2half(o);
        }
    }
}

__device__ __forceinline__ void online4(
    float logit, const float vv[4], float& mx, float& lse, float acc[4]
) {
    float nm = fmaxf(mx, logit);
    float so = (mx == -INFINITY) ? 0.f : exp2f((mx - nm) * kLog2e);
    float e = (logit == -INFINITY) ? 0.f : exp2f((logit - nm) * kLog2e);
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] = acc[i] * so + e * vv[i];
    lse = lse * so + e;
    mx = nm;
}

// ---- Warp-parameterised split-K partial.
// pq_attend_partial_kernel is hard-wired to 4 warps (__launch_bounds__(128,4)), i.e.
// 4 CTAs x 4 warps = 16 warps per SM out of 64 -> 25% occupancy, while its inner work
// is a *random-access gather* of K/S key/value rows whose latency needs occupancy to
// hide.  Measured byte roofline for the gather is ~5 us/layer against ~85 us actual.

// ---------------------------------------------------------------------------
// (opt-in, PQ_HSA_REDUCE_FUSED=1): "last CTA reduces" epilogue for the split-K
// partial kernels.  Every (h,g,s) CTA publishes its partials exactly as before, then
// increments a per-head semaphore; the CTA that observes the final count folds the S
// splits of ALL kG groups of head h (one warp per group) and writes the context row,
// replacing the separate pq_attend_reduce_mw_kernel launch.  The fold arithmetic is
// the p_bg branch of pq_attend_reduce_mw_kernel, verbatim.  Requires bg fused into
// the partial (p_bg slot layout [H,kG,S,D]).  fr_sem == nullptr -> no-op (default).
// ---------------------------------------------------------------------------
template <int NW>
__device__ __forceinline__ void pq_fr_epilogue(
    int h, int S, int D, int L,
    const float* __restrict__ p_fmax, const float* __restrict__ p_flse, const float* __restrict__ p_facc,
    const float* __restrict__ p_emax, const float* __restrict__ p_else, const float* __restrict__ p_eacc,
    const float* __restrict__ p_omax, const float* __restrict__ p_olse,
    const float* __restrict__ p_bg,
    const float* __restrict__ ret_max_in, const float* __restrict__ ret_exp_in,
    int* __restrict__ sem, __half* __restrict__ out, int* s_flag)
{
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        int old = atomicAdd(&sem[h], 1);
        *s_flag = (old == kG * S - 1) ? 1 : 0;
    }
    __syncthreads();
    if (*s_flag == 0) return;
    __threadfence();
    if (threadIdx.x == 0) sem[h] = 0;   // all kG*S CTAs of this head have arrived
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int dbase = lane * 4;
    const bool on = (dbase < D);
    for (int gg = wid; gg < kG; gg += NW) {
        const int rowb = (h * kG + gg) * S;
        float fmax_m = -INFINITY, emax_m = -INFINITY, omax_m = -INFINITY;
        for (int s = 0; s < S; ++s) {
            fmax_m = fmaxf(fmax_m, __ldcg(p_fmax + rowb + s));
            emax_m = fmaxf(emax_m, __ldcg(p_emax + rowb + s));
            omax_m = fmaxf(omax_m, __ldcg(p_omax + rowb + s));
        }
        float flse_m = 0.f, else_m = 0.f, olse_m = 0.f;
        float facc_m[4] = {0, 0, 0, 0}, eacc_m[4] = {0, 0, 0, 0}, bg[4] = {0, 0, 0, 0};
        for (int s = 0; s < S; ++s) {
            const int sl = rowb + s;
            const float fm = __ldcg(p_fmax + sl), em = __ldcg(p_emax + sl), om = __ldcg(p_omax + sl);
            const float fso = (fm == -INFINITY) ? 0.f : exp2f((fm - fmax_m) * kLog2e);
            const float eso = (em == -INFINITY) ? 0.f : exp2f((em - emax_m) * kLog2e);
            const float oso = (om == -INFINITY) ? 0.f : exp2f((om - omax_m) * kLog2e);
            flse_m += __ldcg(p_flse + sl) * fso;
            else_m += __ldcg(p_else + sl) * eso;
            olse_m += __ldcg(p_olse + sl) * oso;
            if (on) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    facc_m[i] += __ldcg(p_facc + (long long)sl * D + dbase + i) * fso;
                    eacc_m[i] += __ldcg(p_eacc + (long long)sl * D + dbase + i) * eso;
                    bg[i] += __ldcg(p_bg + (long long)sl * D + dbase + i);
                }
            }
        }
        const float ret_max = ret_max_in[h * kG + gg];
        const float ret_sum = ret_exp_in[h * kG + gg];
        const float row_max = fmaxf(fmaxf(fmax_m, ret_max), fmaxf(emax_m, omax_m));
        const float full_sum = flse_m * ((fmax_m == -INFINITY) ? 0.f : exp2f((fmax_m - row_max) * kLog2e));
        const float exact_sum = else_m * ((emax_m == -INFINITY) ? 0.f : exp2f((emax_m - row_max) * kLog2e));
        const float old_sum = olse_m * ((omax_m == -INFINITY) ? 0.f : exp2f((omax_m - row_max) * kLog2e));
        const float ret_adj = ret_sum * exp2f((ret_max - row_max) * kLog2e);
        float denom = full_sum + ret_adj - old_sum + exact_sum;
        if (denom < 1e-16f) denom = 1e-16f;
        const float full_scale = (fmax_m == -INFINITY) ? 0.f : exp2f((fmax_m - row_max) * kLog2e) / denom;
        const float exact_scale = (emax_m == -INFINITY) ? 0.f : exp2f((emax_m - row_max) * kLog2e) / denom;
        const float scale_bg = exp2f((ret_max - row_max) * kLog2e);
        if (on) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int d = dbase + i;
                if (d < D) {
                    const float o = facc_m[i] * full_scale + eacc_m[i] * exact_scale
                                  + (bg[i] * scale_bg) / denom;
                    out[(h * kG + gg) * D + d] = __float2half(o);
                }
            }
        }
    }
}

// This template widens the CTA to NW warps (PQ_HSA_CUDA_PARTWARPS) so a CTA covers NW
// tokens per iteration instead of 4.  Same arithmetic, same order within a warp; only
// the cross-warp merge width changes (fp32 re-association, like the multi-warp reduce).
template <int NW>
__device__ __forceinline__ void merge_warp_states_nw(
    float w_max, float w_lse, float acc[4], int dbase,
    float* sm_max, float* sm_lse, float* sm_acc,
    float* out_max, float* out_lse, float out_acc[4]
) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) { sm_max[wid] = w_max; sm_lse[wid] = w_lse; }
#pragma unroll
    for (int i = 0; i < 4; ++i) sm_acc[wid * kDMax + dbase + i] = acc[i];
    __syncthreads();
    float m = sm_max[0];
#pragma unroll
    for (int w = 1; w < NW; ++w) m = fmaxf(m, sm_max[w]);
    float lse = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) out_acc[i] = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) {
        float so = (sm_max[w] == -INFINITY) ? 0.f : exp2f((sm_max[w] - m) * kLog2e);
        lse += sm_lse[w] * so;
#pragma unroll
        for (int i = 0; i < 4; ++i) out_acc[i] += sm_acc[w * kDMax + dbase + i] * so;
    }
    *out_max = m;
    *out_lse = lse;
    __syncthreads();
}

template <int NW, int UNROLL>
__global__ void __launch_bounds__(NW * 32, 2) pq_attend_partial_nw_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ full_k,
    const __half* __restrict__ full_v,
    const float* __restrict__ mask,
    const int32_t* __restrict__ topk_idx,
    const float* __restrict__ topk_val,
    const int64_t* __restrict__ ret_global,
    const __half* __restrict__ sb_k,
    const __half* __restrict__ sb_v,
    float* __restrict__ p_fmax,
    float* __restrict__ p_flse,
    float* __restrict__ p_facc,
    float* __restrict__ p_emax,
    float* __restrict__ p_else,
    float* __restrict__ p_eacc,
    float* __restrict__ p_omax,
    float* __restrict__ p_olse,
    int D, int N, int K, int F, int CAP,
    float scale,
    int mode,
    // BGFUSE: when p_bg != nullptr the background / centroid-correction bracket
    // for this same split is computed here instead of in its own kernel.  The exact
    // loop has already loaded tidx[t] / tval[t] for exactly these t, so the fused form
    // drops one level of the (tidx -> lids -> cents) dependent chain and issues the two
    // extra loads alongside the K/V gathers instead of in a separate grid.
    const int32_t* __restrict__ list_ids,
    const float* __restrict__ list_mass,
    const __half* __restrict__ centroids,
    const float* __restrict__ ret_max_in,
    float* __restrict__ p_bg,
    int L,
    int mask_skip,
    int* __restrict__ fr_sem,
    const float* __restrict__ fr_ret_exp,
    __half* __restrict__ fr_out
) {
    const int h = blockIdx.x;
    const int g = blockIdx.y;
    const int s = blockIdx.z;
    const int S = gridDim.z;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int dbase = lane * 4;
    const bool on = (dbase < D);
    const bool do_v = (mode != 1);
    const bool do_exact = (mode != 2);
    const bool do_full = (mode != 3);
    const int f0 = (F * s) / S;
    const int f1 = (F * (s + 1)) / S;
    const int t0 = (K * s) / S;
    const int t1 = (K * (s + 1)) / S;

    extern __shared__ char dynp[];
    float* sm_max = reinterpret_cast<float*>(dynp);
    float* sm_lse = sm_max + NW;
    float* sm_acc = sm_lse + NW;

    if (do_full && g == 0) {
        float qg[kG][4];
#pragma unroll
        for (int gg = 0; gg < kG; ++gg) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int d = dbase + i;
                qg[gg][i] = (on && d < D) ? __half2float(q[(h * kG + gg) * D + d]) : 0.f;
            }
        }
        float fmax[kG], flse[kG], facc[kG][4];
#pragma unroll
        for (int gg = 0; gg < kG; ++gg) {
            fmax[gg] = -INFINITY; flse[gg] = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) facc[gg][i] = 0.f;
        }
        for (int f = f0 + wid; f < f1; f += NW) {
            // The static full-region buffer is FULL_CAP slots wide but only
            // sink+local are live; the rest carry mask = -inf, so their softmax weight
            // is exactly 0 and they cannot move the running max.  Skipping them is
            // bit-exact and saves the 512 B K/V gather per dead slot.  The branch is
            // warp-uniform (f depends on wid, not lane), so warp_sum stays legal.
            float mkg[kG];
#pragma unroll
            for (int gg = 0; gg < kG; ++gg) mkg[gg] = mask[(h * kG + gg) * F + f];
            if (mask_skip) {
                bool any_live = false;
#pragma unroll
                for (int gg = 0; gg < kG; ++gg) any_live |= (mkg[gg] > -INFINITY);
                if (!any_live) continue;
            }
            float kv[4], vv[4] = {0, 0, 0, 0};
            load4_u2(full_k, static_cast<int64_t>(h) * F + f, D, dbase, on, kv);
            if (do_v) load4_u2(full_v, static_cast<int64_t>(h) * F + f, D, dbase, on, vv);
#pragma unroll
            for (int gg = 0; gg < kG; ++gg) {
                float part = 0.f;
#pragma unroll
                for (int i = 0; i < 4; ++i) part += qg[gg][i] * kv[i];
                float logit = warp_sum(part) * scale + mkg[gg];
                online4(logit, vv, fmax[gg], flse[gg], facc[gg]);
            }
        }
#pragma unroll
        for (int gg = 0; gg < kG; ++gg) {
            float fm, fl, fa[4];
            merge_warp_states_nw<NW>(fmax[gg], flse[gg], facc[gg], dbase, sm_max, sm_lse, sm_acc,
                                     &fm, &fl, fa);
            const int slot = ((h * kG + gg) * S + s);
            if (threadIdx.x == 0) { p_fmax[slot] = fm; p_flse[slot] = fl; }
            if (on) {
#pragma unroll
                for (int i = 0; i < 4; ++i) p_facc[slot * D + dbase + i] = fa[i];
            }
        }
    }

    float exact_max = -INFINITY, exact_lse = 0.f, exact_acc[4] = {0, 0, 0, 0};
    float old_max = -INFINITY, old_lse = 0.f;
    const bool do_bg = (p_bg != nullptr);
    float bgv[4] = {0, 0, 0, 0};
    float bg_ret_max = 0.f;
    const __half* bg_cents = centroids + static_cast<int64_t>(h) * L * D;
    const int32_t* bg_lids = list_ids + h * N;
    if (do_bg) {
        bg_ret_max = ret_max_in[h * kG + g];
        const float* bmass = list_mass + (h * kG + g) * L;
        const int e0 = (L * s) / S;
        const int e1 = (L * (s + 1)) / S;
        for (int ell = e0 + wid; ell < e1; ell += NW) {
            float m = bmass[ell];
            float c4[4];
            load4_u2(bg_cents, static_cast<int64_t>(ell), D, dbase, on, c4);
#pragma unroll
            for (int i = 0; i < 4; ++i) bgv[i] += m * c4[i];
        }
    }
    const int32_t* tidx = topk_idx + (h * kG + g) * K;
    const float* tval = topk_val + (h * kG + g) * K;
    const int64_t* rg = ret_global + h * N;
    float qv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int d = dbase + i;
        qv[i] = (on && d < D) ? __half2float(q[(h * kG + g) * D + d]) : 0.f;
    }
    if (do_exact) {
        // UNROLL independent gathers are issued BEFORE any of them is consumed,
        // so the warp has UNROLL outstanding 256 B random reads instead of one.  The
        // arithmetic and its order are unchanged (the online4 / old_* chains still run
        // strictly in token order), so this is bit-identical to UNROLL=1.
        for (int t = t0 + wid * UNROLL; t < t1; t += NW * UNROLL) {
            int64_t flat[UNROLL];
            float approx[UNROLL];
            bool live[UNROLL];
            float kv[UNROLL][4], vv[UNROLL][4];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                int tt = t + u;
                live[u] = (tt < t1);
                int idx_t = live[u] ? tt : t0;
                int local = tidx[idx_t];
                if (local < 0) local = 0;
                if (local >= N) local = 0;
                approx[u] = tval[idx_t];
                flat[u] = rg[local] + static_cast<int64_t>(h) * CAP;
#pragma unroll
                for (int i = 0; i < 4; ++i) vv[u][i] = 0.f;
            }
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                load4_u2(sb_k, flat[u], D, dbase, on, kv[u]);
                if (do_v) load4_u2(sb_v, flat[u], D, dbase, on, vv[u]);
            }
            if (do_bg) {
                int lid[UNROLL];
#pragma unroll
                for (int u = 0; u < UNROLL; ++u) {
                    int tt = t + u;
                    int loc = tidx[(tt < t1) ? tt : t0];
                    if (loc < 0 || loc >= N) loc = 0;
                    int li = bg_lids[loc];
                    if (li < 0 || li >= L) li = 0;
                    lid[u] = li;
                }
                float bc[UNROLL][4];
#pragma unroll
                for (int u = 0; u < UNROLL; ++u)
                    load4_u2(bg_cents, static_cast<int64_t>(lid[u]), D, dbase, on, bc[u]);
#pragma unroll
                for (int u = 0; u < UNROLL; ++u) {
                    if (!live[u]) continue;
                    float wgt = exp2f((approx[u] - bg_ret_max) * kLog2e);
#pragma unroll
                    for (int i = 0; i < 4; ++i) bgv[i] -= wgt * bc[u][i];
                }
            }
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                float part = 0.f;
#pragma unroll
                for (int i = 0; i < 4; ++i) part += qv[i] * kv[u][i];
                float lg = warp_sum(part) * scale;
                if (!live[u]) continue;
                online4(lg, vv[u], exact_max, exact_lse, exact_acc);
                float om = fmaxf(old_max, approx[u]);
                float oso = (old_max == -INFINITY) ? 0.f : exp2f((old_max - om) * kLog2e);
                old_lse = old_lse * oso + exp2f((approx[u] - om) * kLog2e);
                old_max = om;
            }
        }
    }
    float emax_m, else_m, eacc_m[4];
    merge_warp_states_nw<NW>(exact_max, exact_lse, exact_acc, dbase, sm_max, sm_lse, sm_acc,
                             &emax_m, &else_m, eacc_m);
    if (lane == 0) { sm_max[wid] = old_max; sm_lse[wid] = old_lse; }
    __syncthreads();
    float omax_m = sm_max[0];
#pragma unroll
    for (int w = 1; w < NW; ++w) omax_m = fmaxf(omax_m, sm_max[w]);
    float olse_m = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) {
        float so = (sm_max[w] == -INFINITY) ? 0.f : exp2f((sm_max[w] - omax_m) * kLog2e);
        olse_m += sm_lse[w] * so;
    }
    __syncthreads();
    const int slot = ((h * kG + g) * S + s);
    if (threadIdx.x == 0) {
        p_emax[slot] = emax_m; p_else[slot] = else_m;
        p_omax[slot] = omax_m; p_olse[slot] = olse_m;
    }
    if (on) {
#pragma unroll
        for (int i = 0; i < 4; ++i) p_eacc[slot * D + dbase + i] = eacc_m[i];
    }
    if (do_bg) {
        // sm_acc is [NW][kDMax]; reuse it now that the exact merge is done.
        __syncthreads();
        if (on) {
#pragma unroll
            for (int i = 0; i < 4; ++i) sm_acc[wid * kDMax + dbase + i] = bgv[i];
        }
        __syncthreads();
        if (wid == 0 && on) {
            float* dst = p_bg + slot * D;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float a = 0.f;
#pragma unroll
                for (int w = 0; w < NW; ++w) a += sm_acc[w * kDMax + dbase + i];
                if (dbase + i < D) dst[dbase + i] = a;
            }
        }
    }
    if (fr_sem != nullptr) {
        __shared__ int s_fr_flag;
        pq_fr_epilogue<NW>(h, S, D, L, p_fmax, p_flse, p_facc, p_emax, p_else, p_eacc,
                           p_omax, p_olse, p_bg, ret_max_in, fr_ret_exp, fr_sem, fr_out, &s_fr_flag);
    }
}

// ---------------------------------------------------------------------------
// Paged-KV twin of pq_attend_partial_nw_kernel<NW,UNROLL> (SPLITS>1
// support for PQ_HSA_PAGED_ATTEND=1). pq_attend_partial_nw_kernel above is
// byte-for-byte untouched. Only the exact-gather load (sb_k/sb_v, flat
// [H*CAP,D] addressed by ret_global[local]+h*CAP) is replaced by a read
// straight out of vLLM's own paged kv_cache via this request's block_table,
// using the SAME pq_paged_slot() helper and the SAME "ret_global[local] is
// an ABSOLUTE token position" convention as pq_exact_attend_paged_kernel
// (see the comment block above that kernel). The do_full (sink+local
// full-region) loop, the background/centroid bracket, and every reduction
// are IDENTICAL to pq_attend_partial_nw_kernel -- same value set, same warp
// reduction order, same fp32 accumulation, same outputs (p_fmax/p_flse/...),
// consumed by the SAME (unmodified) pq_attend_bg_kernel/pq_attend_bgmm_kernel
// /pq_attend_reduce_mw_kernel/pq_attend_reduce_kernel downstream -- those
// kernels only touch centroids/list_mass/topk_idx/topk_val plus this
// kernel's partial outputs, never raw K/V, so they need no paged variant.
// ---------------------------------------------------------------------------
template <int NW, int UNROLL>
__global__ void __launch_bounds__(NW * 32, 2) pq_attend_partial_nw_paged_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ full_k,
    const __half* __restrict__ full_v,
    const float* __restrict__ mask,
    const int32_t* __restrict__ topk_idx,
    const float* __restrict__ topk_val,
    const int64_t* __restrict__ ret_global,   // ABSOLUTE token positions
    const __half* __restrict__ kv_k_base,     // vLLM kv_cache[0,...] flat [num_blocks*block_size, num_kv_heads, D]
    const __half* __restrict__ kv_v_base,     // vLLM kv_cache[1,...]
    const int32_t* __restrict__ block_table,  // this request's row, [max_blocks]
    float* __restrict__ p_fmax,
    float* __restrict__ p_flse,
    float* __restrict__ p_facc,
    float* __restrict__ p_emax,
    float* __restrict__ p_else,
    float* __restrict__ p_eacc,
    float* __restrict__ p_omax,
    float* __restrict__ p_olse,
    int D, int N, int K, int F,
    int block_size, int num_kv_heads,
    float scale,
    int mode,
    const int32_t* __restrict__ list_ids,
    const float* __restrict__ list_mass,
    const __half* __restrict__ centroids,
    const float* __restrict__ ret_max_in,
    float* __restrict__ p_bg,
    int L,
    int mask_skip,
    int* __restrict__ fr_sem,
    const float* __restrict__ fr_ret_exp,
    __half* __restrict__ fr_out,
    int kv_layout, int64_t kv_sB, int64_t kv_sH, int64_t kv_sN
) {
    const int h = blockIdx.x;
    const int g = blockIdx.y;
    const int s = blockIdx.z;
    const int S = gridDim.z;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int dbase = lane * 4;
    const bool on = (dbase < D);
    const bool do_v = (mode != 1);
    const bool do_exact = (mode != 2);
    const bool do_full = (mode != 3);
    const int f0 = (F * s) / S;
    const int f1 = (F * (s + 1)) / S;
    const int t0 = (K * s) / S;
    const int t1 = (K * (s + 1)) / S;

    extern __shared__ char dynpp[];
    float* sm_max = reinterpret_cast<float*>(dynpp);
    float* sm_lse = sm_max + NW;
    float* sm_acc = sm_lse + NW;

    if (do_full && g == 0) {
        float qg[kG][4];
#pragma unroll
        for (int gg = 0; gg < kG; ++gg) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int d = dbase + i;
                qg[gg][i] = (on && d < D) ? __half2float(q[(h * kG + gg) * D + d]) : 0.f;
            }
        }
        float fmax[kG], flse[kG], facc[kG][4];
#pragma unroll
        for (int gg = 0; gg < kG; ++gg) {
            fmax[gg] = -INFINITY; flse[gg] = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) facc[gg][i] = 0.f;
        }
        for (int f = f0 + wid; f < f1; f += NW) {
            float mkg[kG];
#pragma unroll
            for (int gg = 0; gg < kG; ++gg) mkg[gg] = mask[(h * kG + gg) * F + f];
            if (mask_skip) {
                bool any_live = false;
#pragma unroll
                for (int gg = 0; gg < kG; ++gg) any_live |= (mkg[gg] > -INFINITY);
                if (!any_live) continue;
            }
            float kv[4], vv[4] = {0, 0, 0, 0};
            load4_u2(full_k, static_cast<int64_t>(h) * F + f, D, dbase, on, kv);
            if (do_v) load4_u2(full_v, static_cast<int64_t>(h) * F + f, D, dbase, on, vv);
#pragma unroll
            for (int gg = 0; gg < kG; ++gg) {
                float part = 0.f;
#pragma unroll
                for (int i = 0; i < 4; ++i) part += qg[gg][i] * kv[i];
                float logit = warp_sum(part) * scale + mkg[gg];
                online4(logit, vv, fmax[gg], flse[gg], facc[gg]);
            }
        }
#pragma unroll
        for (int gg = 0; gg < kG; ++gg) {
            float fm, fl, fa[4];
            merge_warp_states_nw<NW>(fmax[gg], flse[gg], facc[gg], dbase, sm_max, sm_lse, sm_acc,
                                     &fm, &fl, fa);
            const int slot = ((h * kG + gg) * S + s);
            if (threadIdx.x == 0) { p_fmax[slot] = fm; p_flse[slot] = fl; }
            if (on) {
#pragma unroll
                for (int i = 0; i < 4; ++i) p_facc[slot * D + dbase + i] = fa[i];
            }
        }
    }

    float exact_max = -INFINITY, exact_lse = 0.f, exact_acc[4] = {0, 0, 0, 0};
    float old_max = -INFINITY, old_lse = 0.f;
    const bool do_bg = (p_bg != nullptr);
    float bgv[4] = {0, 0, 0, 0};
    float bg_ret_max = 0.f;
    const __half* bg_cents = centroids + static_cast<int64_t>(h) * L * D;
    const int32_t* bg_lids = list_ids + h * N;
    if (do_bg) {
        bg_ret_max = ret_max_in[h * kG + g];
        const float* bmass = list_mass + (h * kG + g) * L;
        const int e0 = (L * s) / S;
        const int e1 = (L * (s + 1)) / S;
        for (int ell = e0 + wid; ell < e1; ell += NW) {
            float m = bmass[ell];
            float c4[4];
            load4_u2(bg_cents, static_cast<int64_t>(ell), D, dbase, on, c4);
#pragma unroll
            for (int i = 0; i < 4; ++i) bgv[i] += m * c4[i];
        }
    }
    const int32_t* tidx = topk_idx + (h * kG + g) * K;
    const float* tval = topk_val + (h * kG + g) * K;
    const int64_t* rg = ret_global + h * N;
    float qv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int d = dbase + i;
        qv[i] = (on && d < D) ? __half2float(q[(h * kG + g) * D + d]) : 0.f;
    }
    if (do_exact) {
        for (int t = t0 + wid * UNROLL; t < t1; t += NW * UNROLL) {
            int64_t rowk[UNROLL], rowv[UNROLL];
            float approx[UNROLL];
            bool live[UNROLL];
            float kv[UNROLL][4], vv[UNROLL][4];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                int tt = t + u;
                live[u] = (tt < t1);
                int idx_t = live[u] ? tt : t0;
                int local = tidx[idx_t];
                if (local < 0) local = 0;
                if (local >= N) local = 0;
                approx[u] = tval[idx_t];
                int64_t pos = rg[local];
                rowk[u] = pq_paged_row(block_table, pos, block_size, num_kv_heads, h, kv_layout, kv_sB, kv_sH, kv_sN);
                rowv[u] = rowk[u];
#pragma unroll
                for (int i = 0; i < 4; ++i) vv[u][i] = 0.f;
            }
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                load4_u2(kv_k_base, rowk[u], D, dbase, on, kv[u]);
                if (do_v) load4_u2(kv_v_base, rowv[u], D, dbase, on, vv[u]);
            }
            if (do_bg) {
                int lid[UNROLL];
#pragma unroll
                for (int u = 0; u < UNROLL; ++u) {
                    int tt = t + u;
                    int loc = tidx[(tt < t1) ? tt : t0];
                    if (loc < 0 || loc >= N) loc = 0;
                    int li = bg_lids[loc];
                    if (li < 0 || li >= L) li = 0;
                    lid[u] = li;
                }
                float bc[UNROLL][4];
#pragma unroll
                for (int u = 0; u < UNROLL; ++u)
                    load4_u2(bg_cents, static_cast<int64_t>(lid[u]), D, dbase, on, bc[u]);
#pragma unroll
                for (int u = 0; u < UNROLL; ++u) {
                    if (!live[u]) continue;
                    float wgt = exp2f((approx[u] - bg_ret_max) * kLog2e);
#pragma unroll
                    for (int i = 0; i < 4; ++i) bgv[i] -= wgt * bc[u][i];
                }
            }
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                float part = 0.f;
#pragma unroll
                for (int i = 0; i < 4; ++i) part += qv[i] * kv[u][i];
                float lg = warp_sum(part) * scale;
                if (!live[u]) continue;
                online4(lg, vv[u], exact_max, exact_lse, exact_acc);
                float om = fmaxf(old_max, approx[u]);
                float oso = (old_max == -INFINITY) ? 0.f : exp2f((old_max - om) * kLog2e);
                old_lse = old_lse * oso + exp2f((approx[u] - om) * kLog2e);
                old_max = om;
            }
        }
    }
    float emax_m, else_m, eacc_m[4];
    merge_warp_states_nw<NW>(exact_max, exact_lse, exact_acc, dbase, sm_max, sm_lse, sm_acc,
                             &emax_m, &else_m, eacc_m);
    if (lane == 0) { sm_max[wid] = old_max; sm_lse[wid] = old_lse; }
    __syncthreads();
    float omax_m = sm_max[0];
#pragma unroll
    for (int w = 1; w < NW; ++w) omax_m = fmaxf(omax_m, sm_max[w]);
    float olse_m = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) {
        float so = (sm_max[w] == -INFINITY) ? 0.f : exp2f((sm_max[w] - omax_m) * kLog2e);
        olse_m += sm_lse[w] * so;
    }
    __syncthreads();
    const int slot = ((h * kG + g) * S + s);
    if (threadIdx.x == 0) {
        p_emax[slot] = emax_m; p_else[slot] = else_m;
        p_omax[slot] = omax_m; p_olse[slot] = olse_m;
    }
    if (on) {
#pragma unroll
        for (int i = 0; i < 4; ++i) p_eacc[slot * D + dbase + i] = eacc_m[i];
    }
    if (do_bg) {
        __syncthreads();
        if (on) {
#pragma unroll
            for (int i = 0; i < 4; ++i) sm_acc[wid * kDMax + dbase + i] = bgv[i];
        }
        __syncthreads();
        if (wid == 0 && on) {
            float* dst = p_bg + slot * D;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float a = 0.f;
#pragma unroll
                for (int w = 0; w < NW; ++w) a += sm_acc[w * kDMax + dbase + i];
                if (dbase + i < D) dst[dbase + i] = a;
            }
        }
    }
    if (fr_sem != nullptr) {
        __shared__ int s_fr_flag;
        pq_fr_epilogue<NW>(h, S, D, L, p_fmax, p_flse, p_facc, p_emax, p_else, p_eacc,
                           p_omax, p_olse, p_bg, ret_max_in, fr_ret_exp, fr_sem, fr_out, &s_fr_flag);
    }
}

// Split-K partial. grid (H,G,S). g==0 does sink/local for all 4 groups.
__global__ void __launch_bounds__(128, 4) pq_attend_partial_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ full_k,
    const __half* __restrict__ full_v,
    const float* __restrict__ mask,
    const int32_t* __restrict__ topk_idx,
    const float* __restrict__ topk_val,
    const int64_t* __restrict__ ret_global,
    const __half* __restrict__ sb_k,
    const __half* __restrict__ sb_v,
    float* __restrict__ p_fmax,
    float* __restrict__ p_flse,
    float* __restrict__ p_facc,
    float* __restrict__ p_emax,
    float* __restrict__ p_else,
    float* __restrict__ p_eacc,
    float* __restrict__ p_omax,
    float* __restrict__ p_olse,
    int D, int N, int K, int F, int CAP,
    float scale,
    int mode
) {
    const int h = blockIdx.x;
    const int g = blockIdx.y;
    const int s = blockIdx.z;
    const int S = gridDim.z;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int dbase = lane * 4;
    const bool on = (dbase < D);
    const bool do_v = (mode != 1);
    const bool do_exact = (mode != 2);
    const bool do_full = (mode != 3);
    const int f0 = (F * s) / S;
    const int f1 = (F * (s + 1)) / S;
    const int t0 = (K * s) / S;
    const int t1 = (K * (s + 1)) / S;

    extern __shared__ char dynp[];
    float* sm_max = reinterpret_cast<float*>(dynp);
    float* sm_lse = sm_max + 4;
    float* sm_acc = sm_lse + 4;

    if (do_full && g == 0) {
        float qg[kG][4];
#pragma unroll
        for (int gg = 0; gg < kG; ++gg) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int d = dbase + i;
                qg[gg][i] = (on && d < D) ? __half2float(q[(h * kG + gg) * D + d]) : 0.f;
            }
        }
        float fmax[kG], flse[kG], facc[kG][4];
#pragma unroll
        for (int gg = 0; gg < kG; ++gg) {
            fmax[gg] = -INFINITY;
            flse[gg] = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) facc[gg][i] = 0.f;
        }
        for (int f = f0 + wid; f < f1; f += 4) {
            float kv[4], vv[4] = {0, 0, 0, 0};
            load4_u2(full_k, static_cast<int64_t>(h) * F + f, D, dbase, on, kv);
            if (do_v) load4_u2(full_v, static_cast<int64_t>(h) * F + f, D, dbase, on, vv);
#pragma unroll
            for (int gg = 0; gg < kG; ++gg) {
                float part = 0.f;
#pragma unroll
                for (int i = 0; i < 4; ++i) part += qg[gg][i] * kv[i];
                float logit = warp_sum(part) * scale + mask[(h * kG + gg) * F + f];
                online4(logit, vv, fmax[gg], flse[gg], facc[gg]);
            }
        }
#pragma unroll
        for (int gg = 0; gg < kG; ++gg) {
            float fm, fl, fa[4];
            merge_warp_states(fmax[gg], flse[gg], facc[gg], dbase, sm_max, sm_lse, sm_acc,
                              &fm, &fl, fa);
            const int slot = ((h * kG + gg) * S + s);
            if (threadIdx.x == 0) {
                p_fmax[slot] = fm;
                p_flse[slot] = fl;
            }
            if (on) {
#pragma unroll
                for (int i = 0; i < 4; ++i) p_facc[slot * D + dbase + i] = fa[i];
            }
        }
    }

    float exact_max = -INFINITY, exact_lse = 0.f, exact_acc[4] = {0, 0, 0, 0};
    float old_max = -INFINITY, old_lse = 0.f;
    const int32_t* tidx = topk_idx + (h * kG + g) * K;
    const float* tval = topk_val + (h * kG + g) * K;
    const int64_t* rg = ret_global + h * N;
    float qv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int d = dbase + i;
        qv[i] = (on && d < D) ? __half2float(q[(h * kG + g) * D + d]) : 0.f;
    }
    if (do_exact) {
        for (int t = t0 + wid; t < t1; t += 4) {
            int local = tidx[t];
            if (local < 0) local = 0;
            if (local >= N) local = 0;
            float approx = tval[t];
            int64_t glob = rg[local];
            int64_t flat = glob + static_cast<int64_t>(h) * CAP;
            float kv[4], vv[4] = {0, 0, 0, 0};
            load4_u2(sb_k, flat, D, dbase, on, kv);
            if (do_v) load4_u2(sb_v, flat, D, dbase, on, vv);
            float part = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) part += qv[i] * kv[i];
            online4(warp_sum(part) * scale, vv, exact_max, exact_lse, exact_acc);
            float om = fmaxf(old_max, approx);
            float oso = (old_max == -INFINITY) ? 0.f : exp2f((old_max - om) * kLog2e);
            old_lse = old_lse * oso + exp2f((approx - om) * kLog2e);
            old_max = om;
        }
    }
    float emax_m, else_m, eacc_m[4];
    merge_warp_states(exact_max, exact_lse, exact_acc, dbase, sm_max, sm_lse, sm_acc,
                      &emax_m, &else_m, eacc_m);
    if (lane == 0) {
        sm_max[wid] = old_max;
        sm_lse[wid] = old_lse;
    }
    __syncthreads();
    float omax_m = sm_max[0];
#pragma unroll
    for (int w = 1; w < 4; ++w) omax_m = fmaxf(omax_m, sm_max[w]);
    float olse_m = 0.f;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        float so = (sm_max[w] == -INFINITY) ? 0.f : exp2f((sm_max[w] - omax_m) * kLog2e);
        olse_m += sm_lse[w] * so;
    }
    __syncthreads();
    const int slot = ((h * kG + g) * S + s);
    if (threadIdx.x == 0) {
        p_emax[slot] = emax_m;
        p_else[slot] = else_m;
        p_omax[slot] = omax_m;
        p_olse[slot] = olse_m;
    }
    if (on) {
#pragma unroll
        for (int i = 0; i < 4; ++i) p_eacc[slot * D + dbase + i] = eacc_m[i];
    }
}

__global__ void __launch_bounds__(128, 4) pq_attend_reduce_kernel(
    const float* __restrict__ p_fmax,
    const float* __restrict__ p_flse,
    const float* __restrict__ p_facc,
    const float* __restrict__ p_emax,
    const float* __restrict__ p_else,
    const float* __restrict__ p_eacc,
    const float* __restrict__ p_omax,
    const float* __restrict__ p_olse,
    const int32_t* __restrict__ topk_idx,
    const float* __restrict__ topk_val,
    const int32_t* __restrict__ list_ids,
    const float* __restrict__ list_mass,
    const __half* __restrict__ centroids,
    const float* __restrict__ row_max_in,
    const float* __restrict__ ret_exp_in,
    __half* __restrict__ out,
    int D, int N, int K, int L, int S
) {
    const int h = blockIdx.x;
    const int g = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int dbase = lane * 4;
    const bool on = (dbase < D);

    auto fold = [&](const float* mx, const float* ls, const float* acc,
                    float& om, float& ol, float oa[4]) {
        om = -INFINITY;
        for (int s = 0; s < S; ++s) om = fmaxf(om, mx[(h * kG + g) * S + s]);
        ol = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) oa[i] = 0.f;
        for (int s = 0; s < S; ++s) {
            int sl = (h * kG + g) * S + s;
            float so = (mx[sl] == -INFINITY) ? 0.f : exp2f((mx[sl] - om) * kLog2e);
            ol += ls[sl] * so;
            if (on) {
#pragma unroll
                for (int i = 0; i < 4; ++i) oa[i] += acc[sl * D + dbase + i] * so;
            }
        }
    };
    float fmax_m, flse_m, facc_m[4];
    float emax_m, else_m, eacc_m[4];
    fold(p_fmax, p_flse, p_facc, fmax_m, flse_m, facc_m);
    fold(p_emax, p_else, p_eacc, emax_m, else_m, eacc_m);
    float omax_m = -INFINITY;
    for (int s = 0; s < S; ++s) omax_m = fmaxf(omax_m, p_omax[(h * kG + g) * S + s]);
    float olse_m = 0.f;
    for (int s = 0; s < S; ++s) {
        int sl = (h * kG + g) * S + s;
        float so = (p_omax[sl] == -INFINITY) ? 0.f : exp2f((p_omax[sl] - omax_m) * kLog2e);
        olse_m += p_olse[sl] * so;
    }

    const int32_t* tidx = topk_idx + (h * kG + g) * K;
    const float* tval = topk_val + (h * kG + g) * K;
    const int32_t* lids = list_ids + h * N;
    float ret_max = row_max_in[h * kG + g];
    float ret_sum = ret_exp_in[h * kG + g];
    float row_max = fmaxf(fmaxf(fmax_m, ret_max), fmaxf(emax_m, omax_m));
    float full_sum = flse_m * ((fmax_m == -INFINITY) ? 0.f : exp2f((fmax_m - row_max) * kLog2e));
    float exact_sum = else_m * ((emax_m == -INFINITY) ? 0.f : exp2f((emax_m - row_max) * kLog2e));
    float old_sum = olse_m * ((omax_m == -INFINITY) ? 0.f : exp2f((omax_m - row_max) * kLog2e));
    float ret_adj = ret_sum * exp2f((ret_max - row_max) * kLog2e);
    float denom = full_sum + ret_adj - old_sum + exact_sum;
    if (denom < 1e-16f) denom = 1e-16f;
    float full_scale = (fmax_m == -INFINITY) ? 0.f : exp2f((fmax_m - row_max) * kLog2e) / denom;
    float exact_scale = (emax_m == -INFINITY) ? 0.f : exp2f((emax_m - row_max) * kLog2e) / denom;

    float bg[4] = {0, 0, 0, 0};
    const float* mass = list_mass + (h * kG + g) * L;
    const __half* cents = centroids + static_cast<int64_t>(h) * L * D;
    float scale_bg = exp2f((ret_max - row_max) * kLog2e);
    for (int ell = 0; ell < L; ++ell) {
        float m = mass[ell] * scale_bg;
        float c4[4];
        load4_u2(cents, static_cast<int64_t>(ell), D, dbase, on, c4);
#pragma unroll
        for (int i = 0; i < 4; ++i) bg[i] += m * c4[i];
    }
    for (int t = 0; t < K; ++t) {
        int local = tidx[t];
        if (local < 0) local = 0;
        if (local >= N) local = 0;
        float approx = tval[t];
        int lid = lids[local];
        if (lid < 0) lid = 0;
        if (lid >= L) lid = 0;
        float old_m = exp2f((approx - row_max) * kLog2e);
        float c4[4];
        load4_u2(cents, static_cast<int64_t>(lid), D, dbase, on, c4);
#pragma unroll
        for (int i = 0; i < 4; ++i) bg[i] -= old_m * c4[i];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int d = dbase + i;
        if (on && d < D) {
            float o = facc_m[i] * full_scale + eacc_m[i] * exact_scale + bg[i] / denom;
            out[(h * kG + g) * D + d] = __float2half(o);
        }
    }
}


// ---- Multi-warp reduce. The original pq_attend_reduce_kernel runs with
// 128 threads but only lane<32 does useful work, and warps 1..3 duplicate warp 0
// bit-for-bit -> the L=512 centroid GEMV + K=1303 correction loop is executed by a
// SINGLE warp per (h,g), i.e. 32 warps total on a 78-SM GPU. This variant splits
// both loops over RW warps and reduces bg[] in shared memory. Numerics: the same
// terms are summed in a different order (fp32), so results differ only by fp
// association.
// ---- Split the reduce kernel's background/correction loops over RS CTAs.
//
// pq_attend_reduce_mw_kernel runs on grid (H, kG) = 32 CTAs.  H20 has 78 SMs, so at
// most 41% of the machine can be busy no matter how many warps a CTA has -- and
// measured that this kernel, not the exact gather, is 79% of pq_exact_attend
// (0.0566 of 0.0716 ms/layer; the gather is only 0.010).  Its two hot loops are
//
//   bg = sum_ell mass[ell]*scale_bg*C[ell]  -  sum_t exp2((approx_t-row_max)*log2e)*C[lid_t]
//
// with L=512 and K=1302 iterations.  Both depend on row_max / scale_bg, which are only
// known after the split-K fold -- an apparent serial dependency.  It is removable:
// scale_bg == exp2((ret_max - row_max)*log2e) and, because every approx_t is a
// retrieval score and ret_max is their row max, approx_t <= ret_max, so
//
//   bg = scale_bg * ( sum_ell mass[ell]*C[ell] - sum_t exp2((approx_t-ret_max)*log2e)*C[lid_t] )
//
// The bracket depends only on inputs that are ready before the partial kernel, so it
// can be computed by its own grid (H, kG, RS) and folded in afterwards.  Referencing
// the exponent to ret_max instead of row_max is also better conditioned (every term is
// <= 1) rather than worse.  This is an exact refactor up to fp32 rounding.
// ---- The background term is a tiny GEMM in LIST space, not a gather.
//
// The formulation touches one centroid ROW per ell AND one per selected
// token:  bg = sum_ell mass[ell]*C[ell] - sum_t w_t*C[lid_t], i.e. H*kG*(L+K) = 58048
// row reads of 256 B = 14.9 MB/layer, of which the K=1302 per (h,g) are a random
// gather.  But every token t lands in exactly one list lid_t, so the second sum can be
// binned first:
//
//   corr[g][ell] = sum_{t: lid_t == ell} w_t          (K scalars -> L=512 bins)
//   bg[g]        = sum_ell (mass[g][ell] - corr[g][ell]) * C[ell]
//
// which is one [kG,L] x [L,D] GEMM per head.  The centroid table is now read exactly
// once per head (8 x 512 x 256 B = 1 MB/layer instead of 14.9 MB) and the random
// gather disappears entirely.  Same sum, regrouped: an exact refactor up to fp32
// rounding (and better conditioned -- the mass/correction cancellation now happens in
// list space where both terms are O(list mass), before being applied to C).
//
// Stage 1: bin K tokens into L smem bins per (h,g), emit mcorr[H,kG,L].
__global__ void pq_attend_corr_kernel(
    const int32_t* __restrict__ topk_idx,
    const float* __restrict__ topk_val,
    const int32_t* __restrict__ list_ids,
    const float* __restrict__ list_mass,
    const float* __restrict__ ret_max_in,
    float* __restrict__ mcorr,        // [H, kG, L]
    int D, int N, int K, int L
) {
    extern __shared__ float sbin[];
    const int h = blockIdx.x;
    const int g = blockIdx.y;
    const int nt = blockDim.x;
    for (int i = threadIdx.x; i < L; i += nt) sbin[i] = 0.f;
    __syncthreads();
    const float ret_max = ret_max_in[h * kG + g];
    const int32_t* tidx = topk_idx + (h * kG + g) * K;
    const float* tval = topk_val + (h * kG + g) * K;
    const int32_t* lids = list_ids + h * N;
    for (int t = threadIdx.x; t < K; t += nt) {
        int local = tidx[t];
        if (local < 0 || local >= N) local = 0;
        int lid = lids[local];
        if (lid < 0 || lid >= L) lid = 0;
        atomicAdd(&sbin[lid], exp2f((tval[t] - ret_max) * kLog2e));
    }
    __syncthreads();
    const float* mass = list_mass + (h * kG + g) * L;
    float* dst = mcorr + (h * kG + g) * L;
    for (int i = threadIdx.x; i < L; i += nt) dst[i] = mass[i] - sbin[i];
}

// Stage 2: bg[h][g][rs] = sum_{ell in split rs} mcorr[h][g][ell] * C[h][ell].
// One CTA covers all kG groups for a head, so C[ell] is loaded once, not kG times.
template <int NW>
__global__ void pq_attend_bgmm_kernel(
    const float* __restrict__ mcorr,   // [H, kG, L]
    const __half* __restrict__ centroids,
    float* __restrict__ p_bg,          // [H, kG, RS, D]
    int D, int L, int RS
) {
    extern __shared__ float shmm[];    // [NW * kG * kDMax]
    const int h = blockIdx.x;
    const int rs = blockIdx.y;
    const int w = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int dbase = lane * 4;
    const bool on = (dbase < D);
    const int e0 = (L * rs) / RS;
    const int e1 = (L * (rs + 1)) / RS;
    const __half* cents = centroids + static_cast<int64_t>(h) * L * D;

    float acc[kG][4];
#pragma unroll
    for (int g = 0; g < kG; ++g)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[g][i] = 0.f;
    for (int ell = e0 + w; ell < e1; ell += NW) {
        float c4[4];
        load4_u2(cents, static_cast<int64_t>(ell), D, dbase, on, c4);
#pragma unroll
        for (int g = 0; g < kG; ++g) {
            float m = mcorr[(h * kG + g) * L + ell];
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[g][i] += m * c4[i];
        }
    }
    if (on) {
#pragma unroll
        for (int g = 0; g < kG; ++g)
#pragma unroll
            for (int i = 0; i < 4; ++i) shmm[(w * kG + g) * kDMax + dbase + i] = acc[g][i];
    }
    __syncthreads();
    if (w == 0 && on) {
#pragma unroll
        for (int g = 0; g < kG; ++g) {
            float* dst = p_bg + (static_cast<int64_t>((h * kG + g)) * RS + rs) * D;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float a = 0.f;
                for (int ww = 0; ww < NW; ++ww) a += shmm[(ww * kG + g) * kDMax + dbase + i];
                if (dbase + i < D) dst[dbase + i] = a;
            }
        }
    }
}

template <int RW, int BU>
__global__ void pq_attend_bg_kernel(
    const int32_t* __restrict__ topk_idx,
    const float* __restrict__ topk_val,
    const int32_t* __restrict__ list_ids,
    const float* __restrict__ list_mass,
    const __half* __restrict__ centroids,
    const float* __restrict__ ret_max_in,
    float* __restrict__ p_bg,          // [H, kG, RS, D]
    int D, int N, int K, int L, int RS
) {
    extern __shared__ float shbg[];
    const int h = blockIdx.x;
    const int g = blockIdx.y;
    const int rs = blockIdx.z;
    const int w = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int dbase = lane * 4;
    const bool on = (dbase < D);

    const int ell0 = (L * rs) / RS;
    const int ell1 = (L * (rs + 1)) / RS;
    const int t0 = (K * rs) / RS;
    const int t1 = (K * (rs + 1)) / RS;

    const float ret_max = ret_max_in[h * kG + g];
    const float* mass = list_mass + (h * kG + g) * L;
    const __half* cents = centroids + static_cast<int64_t>(h) * L * D;

    float bg[4] = {0, 0, 0, 0};
    for (int ell = ell0 + w; ell < ell1; ell += RW) {
        float m = mass[ell];
        float c4[4];
        load4_u2(cents, static_cast<int64_t>(ell), D, dbase, on, c4);
#pragma unroll
        for (int i = 0; i < 4; ++i) bg[i] += m * c4[i];
    }
    const int32_t* tidx = topk_idx + (h * kG + g) * K;
    const float* tval = topk_val + (h * kG + g) * K;
    const int32_t* lids = list_ids + h * N;
    // Tidx[t] -> lids[local] -> cents[lid] is a 3-deep dependent load chain, and
    // with RS splits each warp only runs a handful of iterations, so the whole kernel is
    // that latency times the iteration count.  BU independent chains are advanced in
    // lockstep so BU of them are outstanding at once.  Arithmetic order per chain is
    // unchanged; the accumulation order across t changes only for BU > 1 (fp32
    // re-association, same magnitude as PARTUNROLL in the partial kernel).
    for (int t = t0 + w * BU; t < t1; t += RW * BU) {
        int local[BU];
        float approx[BU];
        bool live[BU];
#pragma unroll
        for (int u = 0; u < BU; ++u) {
            int tt = t + u;
            live[u] = (tt < t1);
            int ii = live[u] ? tt : t0;
            int lc = tidx[ii];
            if (lc < 0 || lc >= N) lc = 0;
            local[u] = lc;
            approx[u] = tval[ii];
        }
        int lid[BU];
#pragma unroll
        for (int u = 0; u < BU; ++u) {
            int li = lids[local[u]];
            if (li < 0 || li >= L) li = 0;
            lid[u] = li;
        }
        float c4[BU][4];
#pragma unroll
        for (int u = 0; u < BU; ++u) load4_u2(cents, static_cast<int64_t>(lid[u]), D, dbase, on, c4[u]);
#pragma unroll
        for (int u = 0; u < BU; ++u) {
            if (!live[u]) continue;
            float old_m = exp2f((approx[u] - ret_max) * kLog2e);
#pragma unroll
            for (int i = 0; i < 4; ++i) bg[i] -= old_m * c4[u][i];
        }
    }
    if (on) {
#pragma unroll
        for (int i = 0; i < 4; ++i) shbg[w * kDMax + dbase + i] = bg[i];
    }
    __syncthreads();
    if (w == 0 && on) {
        float* dst = p_bg + (static_cast<int64_t>((h * kG + g)) * RS + rs) * D;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float acc = 0.f;
#pragma unroll
            for (int ww = 0; ww < RW; ++ww) acc += shbg[ww * kDMax + dbase + i];
            if (dbase + i < D) dst[dbase + i] = acc;
        }
    }
}

__global__ void pq_attend_reduce_mw_kernel(
    const float* __restrict__ p_fmax,
    const float* __restrict__ p_flse,
    const float* __restrict__ p_facc,
    const float* __restrict__ p_emax,
    const float* __restrict__ p_else,
    const float* __restrict__ p_eacc,
    const float* __restrict__ p_omax,
    const float* __restrict__ p_olse,
    const int32_t* __restrict__ topk_idx,
    const float* __restrict__ topk_val,
    const int32_t* __restrict__ list_ids,
    const float* __restrict__ list_mass,
    const __half* __restrict__ centroids,
    const float* __restrict__ row_max_in,
    const float* __restrict__ ret_exp_in,
    __half* __restrict__ out,
    int D, int N, int K, int L, int S, int RW,
    const float* __restrict__ p_bg_in,   // Precomputed [H,kG,RS,D] bracket, or null
    int RS_in
) {
    extern __shared__ float sh[];
    // layout: [RW*128 bg partials][128 facc][128 eacc][8 scalars]
    float* s_bg  = sh;
    float* s_fa  = sh + RW * 128;
    float* s_ea  = s_fa + 128;
    float* s_sc  = s_ea + 128;   // 8 scalars

    const int h = blockIdx.x;
    const int g = blockIdx.y;
    const int w = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int dbase = lane * 4;
    const bool on = (dbase < D);

    if (w == 0) {
        auto fold = [&](const float* mx, const float* ls, const float* acc,
                        float& om, float& ol, float oa[4]) {
            om = -INFINITY;
            for (int s = 0; s < S; ++s) om = fmaxf(om, mx[(h * kG + g) * S + s]);
            ol = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) oa[i] = 0.f;
            for (int s = 0; s < S; ++s) {
                int sl = (h * kG + g) * S + s;
                float so = (mx[sl] == -INFINITY) ? 0.f : exp2f((mx[sl] - om) * kLog2e);
                ol += ls[sl] * so;
                if (on) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) oa[i] += acc[sl * D + dbase + i] * so;
                }
            }
        };
        float fmax_m, flse_m, facc_m[4];
        float emax_m, else_m, eacc_m[4];
        fold(p_fmax, p_flse, p_facc, fmax_m, flse_m, facc_m);
        fold(p_emax, p_else, p_eacc, emax_m, else_m, eacc_m);
        float omax_m = -INFINITY;
        for (int s = 0; s < S; ++s) omax_m = fmaxf(omax_m, p_omax[(h * kG + g) * S + s]);
        float olse_m = 0.f;
        for (int s = 0; s < S; ++s) {
            int sl = (h * kG + g) * S + s;
            float so = (p_omax[sl] == -INFINITY) ? 0.f : exp2f((p_omax[sl] - omax_m) * kLog2e);
            olse_m += p_olse[sl] * so;
        }
        float ret_max = row_max_in[h * kG + g];
        float ret_sum = ret_exp_in[h * kG + g];
        float row_max = fmaxf(fmaxf(fmax_m, ret_max), fmaxf(emax_m, omax_m));
        float full_sum = flse_m * ((fmax_m == -INFINITY) ? 0.f : exp2f((fmax_m - row_max) * kLog2e));
        float exact_sum = else_m * ((emax_m == -INFINITY) ? 0.f : exp2f((emax_m - row_max) * kLog2e));
        float old_sum = olse_m * ((omax_m == -INFINITY) ? 0.f : exp2f((omax_m - row_max) * kLog2e));
        float ret_adj = ret_sum * exp2f((ret_max - row_max) * kLog2e);
        float denom = full_sum + ret_adj - old_sum + exact_sum;
        if (denom < 1e-16f) denom = 1e-16f;
        if (on) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { s_fa[dbase + i] = facc_m[i]; s_ea[dbase + i] = eacc_m[i]; }
        }
        if (lane == 0) {
            s_sc[0] = row_max;
            s_sc[1] = denom;
            s_sc[2] = (fmax_m == -INFINITY) ? 0.f : exp2f((fmax_m - row_max) * kLog2e) / denom;
            s_sc[3] = (emax_m == -INFINITY) ? 0.f : exp2f((emax_m - row_max) * kLog2e) / denom;
            s_sc[4] = exp2f((ret_max - row_max) * kLog2e);
        }
    }
    __syncthreads();

    const float row_max = s_sc[0];
    const float denom = s_sc[1];
    const float full_scale = s_sc[2];
    const float exact_scale = s_sc[3];
    const float scale_bg = s_sc[4];

    float bg[4] = {0, 0, 0, 0};
    if (p_bg_in != nullptr) {
        // The bracket was computed by pq_attend_bg_kernel on grid (H,kG,RS).
        // Only warp 0 has anything to do; RS is small (<=16) so this is a few loads.
        if (w == 0 && on) {
            const float* src = p_bg_in + static_cast<int64_t>(h * kG + g) * RS_in * D;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float acc = 0.f;
                for (int rs = 0; rs < RS_in; ++rs) acc += src[rs * D + dbase + i];
                int d = dbase + i;
                if (d < D) {
                    float o = s_fa[d] * full_scale + s_ea[d] * exact_scale
                            + (acc * scale_bg) / denom;
                    out[(h * kG + g) * D + d] = __float2half(o);
                }
            }
        }
        return;
    }
    const float* mass = list_mass + (h * kG + g) * L;
    const __half* cents = centroids + static_cast<int64_t>(h) * L * D;
    for (int ell = w; ell < L; ell += RW) {
        float m = mass[ell] * scale_bg;
        float c4[4];
        load4_u2(cents, static_cast<int64_t>(ell), D, dbase, on, c4);
#pragma unroll
        for (int i = 0; i < 4; ++i) bg[i] += m * c4[i];
    }
    const int32_t* tidx = topk_idx + (h * kG + g) * K;
    const float* tval = topk_val + (h * kG + g) * K;
    const int32_t* lids = list_ids + h * N;
    for (int t = w; t < K; t += RW) {
        int local = tidx[t];
        if (local < 0) local = 0;
        if (local >= N) local = 0;
        float approx = tval[t];
        int lid = lids[local];
        if (lid < 0) lid = 0;
        if (lid >= L) lid = 0;
        float old_m = exp2f((approx - row_max) * kLog2e);
        float c4[4];
        load4_u2(cents, static_cast<int64_t>(lid), D, dbase, on, c4);
#pragma unroll
        for (int i = 0; i < 4; ++i) bg[i] -= old_m * c4[i];
    }
    if (on) {
#pragma unroll
        for (int i = 0; i < 4; ++i) s_bg[w * 128 + dbase + i] = bg[i];
    }
    __syncthreads();

    if (w == 0 && on) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float acc = 0.f;
            for (int ww = 0; ww < RW; ++ww) acc += s_bg[ww * 128 + dbase + i];
            int d = dbase + i;
            if (d < D) {
                float o = s_fa[d] * full_scale + s_ea[d] * exact_scale + acc / denom;
                out[(h * kG + g) * D + d] = __float2half(o);
            }
        }
    }
}

}  // namespace

// ---- host launchers ----

static int resolve_C(int N, int64_t num_ctas) {
    int C = choose_ctas_host(N);
    if (num_ctas > 0) C = std::min(64, std::max(1, static_cast<int>(num_ctas)));
    return C;
}

static int scan_smem_bytes() {
    constexpr int kHalf = 2;
    return (kPairs * 256 * kG + kLMax * kG + kMaxTile * kG) * kHalf
        + kMaxTile * static_cast<int>(sizeof(uint16_t))
        + (kG * 256 * 2) * static_cast<int>(sizeof(int))
        + (kG * kLMax) * static_cast<int>(sizeof(float))
        + 32 * static_cast<int>(sizeof(float))
        + (kG * 8) * static_cast<int>(sizeof(int))
        + (kG * 2) * static_cast<int>(sizeof(float))
        + 256;
}

std::vector<torch::Tensor> pq_scan_only_cuda(
    torch::Tensor packed_u32,
    torch::Tensor pair_lut,
    torch::Tensor list_ids,
    torch::Tensor list_scores_t,
    torch::Tensor token_scale,
    int64_t k,
    int64_t num_ctas,
    int64_t mass_mode,
    torch::Tensor score_out
) {
    TORCH_CHECK(packed_u32.is_cuda() && packed_u32.dtype() == torch::kInt32);
    TORCH_CHECK(pair_lut.scalar_type() == torch::kFloat16);
    const int H = static_cast<int>(packed_u32.size(0));
    const int N = static_cast<int>(packed_u32.size(1));
    const int L = static_cast<int>(list_scores_t.size(1));
    TORCH_CHECK(L <= kLMax, "L exceeds kLMax");
    const int K = std::min(std::max(static_cast<int>(k), 1), N);
    const int C = resolve_C(N, num_ctas);
    auto opts_i = list_ids.options().dtype(torch::kInt32);
    auto opts_f = packed_u32.options().dtype(torch::kFloat32);
    auto cand_idx = torch::empty({H, C, kG, K}, opts_i);
    auto cand_val = torch::empty({H, C, kG, K}, opts_f);
    auto bmax = torch::empty({H, C, kG}, opts_f);
    auto bexp = torch::empty({H, C, kG}, opts_f);
    auto lpart = torch::zeros({H, C, kG, kLMax}, opts_f);
    auto clocks = torch::zeros({8}, packed_u32.options().dtype(torch::kInt64));
    const bool has_scale = token_scale.defined() && token_scale.numel() > 0;
    __half* sp = nullptr;
    if (score_out.defined() && score_out.numel() > 0) {
        TORCH_CHECK(score_out.scalar_type() == torch::kFloat16);
        sp = reinterpret_cast<__half*>(score_out.data_ptr<at::Half>());
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    static bool attrs = false;
    if (!attrs) {
        cudaError_t ae = cudaFuncSetAttribute(
            pq_scan_select_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
        TORCH_CHECK(ae == cudaSuccess, "cudaFuncSetAttribute: ", cudaGetErrorString(ae));
        attrs = true;
    }
    const int smem = scan_smem_bytes();
    TORCH_CHECK(smem <= 227 * 1024, "scan-select smem too large: ", smem);
    dim3 grid(C, H);
    pq_scan_select_kernel<<<grid, kBlock, smem, stream>>>(
        reinterpret_cast<const uint32_t*>(packed_u32.data_ptr<int32_t>()),
        list_ids.data_ptr<int32_t>(),
        reinterpret_cast<const __half*>(pair_lut.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(list_scores_t.data_ptr<at::Half>()),
        has_scale ? reinterpret_cast<const __half*>(token_scale.data_ptr<at::Half>()) : nullptr,
        cand_idx.data_ptr<int32_t>(),
        cand_val.data_ptr<float>(),
        bmax.data_ptr<float>(),
        bexp.data_ptr<float>(),
        lpart.data_ptr<float>(),
        sp,
        clocks.data_ptr<int64_t>(),
        N, C, K, L,
        static_cast<int>(packed_u32.stride(0)),
        static_cast<int>(list_ids.stride(0)),
        has_scale ? static_cast<int>(token_scale.stride(0)) : 0,
        has_scale ? 1 : 0,
        static_cast<int>(mass_mode)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {cand_idx, cand_val, bmax, bexp, lpart, clocks};
}

std::vector<torch::Tensor> pq_merge_only_cuda(
    torch::Tensor cand_idx,
    torch::Tensor cand_val,
    torch::Tensor bmax,
    torch::Tensor bexp,
    torch::Tensor lpart,
    int64_t L,
    int64_t merge_threads,
    int64_t do_mass
) {
    const int H = static_cast<int>(cand_idx.size(0));
    const int C = static_cast<int>(cand_idx.size(1));
    const int K = static_cast<int>(cand_idx.size(3));
    auto opts_i = cand_idx.options();
    auto opts_f = cand_val.options();
    auto out_idx = torch::empty({H, kG, K}, opts_i);
    auto out_val = torch::empty({H, kG, K}, opts_f);
    auto row_max = torch::empty({H, kG}, opts_f);
    auto ret_exp = torch::empty({H, kG}, opts_f);
    auto list_mass = torch::empty({H, kG, static_cast<int>(L)}, opts_f);
    int th = (merge_threads >= 1024) ? 1024 : 256;
    dim3 g2(H, kG);
    pq_merge_topk_kernel<<<g2, th, 0, at::cuda::getCurrentCUDAStream()>>>(
        cand_idx.data_ptr<int32_t>(), cand_val.data_ptr<float>(),
        out_idx.data_ptr<int32_t>(), out_val.data_ptr<float>(),
        bmax.data_ptr<float>(), bexp.data_ptr<float>(), lpart.data_ptr<float>(),
        row_max.data_ptr<float>(), ret_exp.data_ptr<float>(), list_mass.data_ptr<float>(),
        C, K, static_cast<int>(L), static_cast<int>(do_mass)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out_val, out_idx, row_max, list_mass, ret_exp};
}

std::vector<torch::Tensor> pq_scan_select_cuda(
    torch::Tensor packed_u32,
    torch::Tensor pair_lut,
    torch::Tensor list_ids,
    torch::Tensor list_scores_t,
    torch::Tensor token_scale,
    int64_t k,
    int64_t num_ctas,
    int64_t mass_mode,
    int64_t merge_threads,
    torch::Tensor score_out
) {
    auto scan = pq_scan_only_cuda(
        packed_u32, pair_lut, list_ids, list_scores_t, token_scale,
        k, num_ctas, mass_mode, score_out
    );
    const int L = static_cast<int>(list_scores_t.size(1));
    auto mer = pq_merge_only_cuda(
        scan[0], scan[1], scan[2], scan[3], scan[4], L, merge_threads,
        mass_mode == 0 ? 1 : 0
    );
    return {mer[0], mer[1], mer[2], mer[3], mer[4], scan[4], scan[2], scan[3], scan[5]};
}

torch::Tensor pq_exact_attend_cuda(
    torch::Tensor q,
    torch::Tensor full_k,
    torch::Tensor full_v,
    torch::Tensor mask,
    torch::Tensor topk_idx,
    torch::Tensor topk_val,
    torch::Tensor ret_global,
    torch::Tensor sb_k,
    torch::Tensor sb_v,
    torch::Tensor list_ids,
    torch::Tensor list_mass,
    torch::Tensor centroids,
    torch::Tensor row_max,
    torch::Tensor ret_exp,
    int64_t cap,
    double scale,
    int64_t n_splits,
    int64_t mode
) {
    const int D = static_cast<int>(q.size(2));
    const int N = static_cast<int>(ret_global.size(1));
    const int K = static_cast<int>(topk_idx.size(2));
    const int F = static_cast<int>(full_k.size(1));
    const int L = static_cast<int>(centroids.size(1));
    const int H = static_cast<int>(q.size(0));
    auto out = torch::empty({H, kG, D}, q.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int attend_smem = (8 + 4 * kDMax) * sizeof(float);
    int S = static_cast<int>(n_splits);
    if (S < 1) S = 1;
    if (S > 32) S = 32;
    int md = static_cast<int>(mode);

    if (S <= 1) {
        dim3 grid(H, kG);
        pq_exact_attend_kernel<<<grid, kDMax, attend_smem, stream>>>(
            reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(full_k.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(full_v.data_ptr<at::Half>()),
            mask.data_ptr<float>(),
            topk_idx.data_ptr<int32_t>(),
            topk_val.data_ptr<float>(),
            ret_global.data_ptr<int64_t>(),
            reinterpret_cast<const __half*>(sb_k.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(sb_v.data_ptr<at::Half>()),
            list_ids.data_ptr<int32_t>(),
            list_mass.data_ptr<float>(),
            reinterpret_cast<const __half*>(centroids.data_ptr<at::Half>()),
            row_max.data_ptr<float>(),
            ret_exp.data_ptr<float>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            D, N, K, F, L, static_cast<int>(cap),
            static_cast<float>(scale), md
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }

    struct WS {
        int H = 0, S = 0, D = 0;
        torch::Tensor p_fmax, p_flse, p_facc, p_emax, p_else, p_eacc, p_omax, p_olse;
        int RS = 0;
        torch::Tensor p_bg;    // [H, kG, RS, D] float32
        torch::Tensor mcorr;   // [H, kG, L] float32
        torch::Tensor sem;     // [H] int32 per-head arrival counter
    };
    static WS ws;
    auto opts_f = q.options().dtype(torch::kFloat32);
    if (ws.H != H || ws.S != S || ws.D != D || !ws.p_fmax.defined()
        || ws.p_fmax.device() != q.device()) {
        ws.H = H; ws.S = S; ws.D = D;
        ws.p_fmax = torch::empty({H, kG, S}, opts_f);
        ws.p_flse = torch::empty({H, kG, S}, opts_f);
        ws.p_facc = torch::empty({H, kG, S, D}, opts_f);
        ws.p_emax = torch::empty({H, kG, S}, opts_f);
        ws.p_else = torch::empty({H, kG, S}, opts_f);
        ws.p_eacc = torch::empty({H, kG, S, D}, opts_f);
        ws.p_omax = torch::empty({H, kG, S}, opts_f);
        ws.p_olse = torch::empty({H, kG, S}, opts_f);
    }
    // g==0 overwrites all groups' full partials; exact is always written.
    auto& p_fmax = ws.p_fmax;
    auto& p_flse = ws.p_flse;
    auto& p_facc = ws.p_facc;
    auto& p_emax = ws.p_emax;
    auto& p_else = ws.p_else;
    auto& p_eacc = ws.p_eacc;
    auto& p_omax = ws.p_omax;
    auto& p_olse = ws.p_olse;

    dim3 grid(H, kG, S);
    // Background/correction split settings are read here because BGFUSE folds
    // them into the partial kernel, which launches first.
    int RS = 0, BW = 8, BU = 1, BGMM = 1, BGFUSE = 0;
    {
        const char* e = std::getenv("PQ_HSA_CUDA_BGSPLITS");
        if (e) { RS = atoi(e); }
        if (RS < 0) RS = 0;
        if (RS > 32) RS = 32;
        const char* e2 = std::getenv("PQ_HSA_CUDA_BGWARPS");
        if (e2) { BW = atoi(e2); }
        if (BW != 4 && BW != 8 && BW != 16 && BW != 32) BW = 8;
        const char* e3 = std::getenv("PQ_HSA_CUDA_BGUNROLL");
        if (e3) { BU = atoi(e3); }
        if (BU != 1 && BU != 2 && BU != 4 && BU != 8) BU = 1;
        const char* e4 = std::getenv("PQ_HSA_CUDA_BGMM");
        if (e4) { BGMM = atoi(e4); }
        const char* e5 = std::getenv("PQ_HSA_CUDA_BGFUSE");
        if (e5) { BGFUSE = atoi(e5); }
    }
    // BGFUSE reuses the partial kernel's own (H,kG,S) split, so RS is forced to S.
    if (BGFUSE && RS >= 1) { RS = S; BGMM = 0; }
    if (RS >= 1) {
        if (ws.RS != RS || !ws.p_bg.defined() || ws.p_bg.size(0) != H
            || ws.p_bg.size(3) != D || ws.p_bg.device() != q.device()) {
            ws.RS = RS;
            ws.p_bg = torch::empty({H, kG, RS, D}, opts_f);
        }
    }
    { const char* e = std::getenv("PQ_HSA_REDUCE_FUSED"); if (e && atoi(e) && S > 1) { BGFUSE = 1; RS = S; BGMM = 0;
        if (ws.RS != RS || !ws.p_bg.defined() || ws.p_bg.size(0) != H || ws.p_bg.size(3) != D || ws.p_bg.device() != q.device()) { ws.RS = RS; ws.p_bg = torch::empty({H, kG, RS, D}, opts_f); } } }
    const bool bg_fused = (BGFUSE && RS >= 1);
    int PW = 4;   // Warps per partial CTA; 4 = the original kernel
    {
        const char* e = std::getenv("PQ_HSA_CUDA_PARTWARPS");
        if (e) { PW = atoi(e); }
        if (PW != 4 && PW != 8 && PW != 16 && PW != 32) PW = 4;
    }
    int PU = 1;   // Gathers in flight per warp in the partial kernel
    {
        const char* e = std::getenv("PQ_HSA_CUDA_PARTUNROLL");
        if (e) { PU = atoi(e); }
        if (PU != 1 && PU != 2 && PU != 4 && PU != 8) PU = 1;
    }
    if (bg_fused && PW == 4 && PU == 1) { PU = 1; PW = 4; }   // still the nw template
    int MASKSKIP = 0;   // Skip dead full-region slots (bit-exact). Default OFF.
    { const char* e = std::getenv("PQ_HSA_MASKSKIP"); if (e) MASKSKIP = atoi(e); }
    // Fold the split reduce into the partial kernel (last CTA per head).  Needs the
    // background bracket inside the partial too (bg fused, RS=S), so force that layout.
    int FR = 0;
    { const char* e = std::getenv("PQ_HSA_REDUCE_FUSED"); if (e) FR = atoi(e); }
    if (FR && S > 1) {
        RS = S; BGMM = 0;
        if (ws.RS != RS || !ws.p_bg.defined() || ws.p_bg.size(0) != H
            || ws.p_bg.size(3) != D || ws.p_bg.device() != q.device()) {
            ws.RS = RS;
            ws.p_bg = torch::empty({H, kG, RS, D}, opts_f);
        }
        if (!ws.sem.defined() || ws.sem.size(0) != H || ws.sem.device() != q.device()) {
            ws.sem = torch::zeros({H}, q.options().dtype(torch::kInt32));
        }
    } else { FR = 0; }
    const bool fr_on = (FR != 0);
    int* fr_sem_ptr = fr_on ? ws.sem.data_ptr<int>() : nullptr;
    const float* fr_ret_exp_ptr = fr_on ? ret_exp.data_ptr<float>() : nullptr;
    __half* fr_out_ptr = fr_on ? reinterpret_cast<__half*>(out.data_ptr<at::Half>()) : nullptr;
    if (PW != 4 || PU != 1 || bg_fused) {

        const int part_smem = (2 * PW + PW * kDMax) * sizeof(float);
#define PQ_LAUNCH_PARTIAL_NW(NWV, UV)                                                  \
        pq_attend_partial_nw_kernel<NWV, UV><<<grid, NWV * 32, part_smem, stream>>>(   \
            reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),                   \
            reinterpret_cast<const __half*>(full_k.data_ptr<at::Half>()),              \
            reinterpret_cast<const __half*>(full_v.data_ptr<at::Half>()),              \
            mask.data_ptr<float>(),                                                    \
            topk_idx.data_ptr<int32_t>(),                                              \
            topk_val.data_ptr<float>(),                                                \
            ret_global.data_ptr<int64_t>(),                                            \
            reinterpret_cast<const __half*>(sb_k.data_ptr<at::Half>()),                \
            reinterpret_cast<const __half*>(sb_v.data_ptr<at::Half>()),                \
            p_fmax.data_ptr<float>(), p_flse.data_ptr<float>(), p_facc.data_ptr<float>(), \
            p_emax.data_ptr<float>(), p_else.data_ptr<float>(), p_eacc.data_ptr<float>(), \
            p_omax.data_ptr<float>(), p_olse.data_ptr<float>(),                        \
            D, N, K, F, static_cast<int>(cap), static_cast<float>(scale), md,           \
            list_ids.data_ptr<int32_t>(), list_mass.data_ptr<float>(),                 \
            reinterpret_cast<const __half*>(centroids.data_ptr<at::Half>()),           \
            row_max.data_ptr<float>(),                                                 \
            bg_fused ? ws.p_bg.data_ptr<float>() : nullptr, L, MASKSKIP,                 \
            fr_sem_ptr, fr_ret_exp_ptr, fr_out_ptr)
#define PQ_DISPATCH_U(NWV)                                                             \
        do {                                                                           \
            if (PU == 1)      { PQ_LAUNCH_PARTIAL_NW(NWV, 1); }                        \
            else if (PU == 2) { PQ_LAUNCH_PARTIAL_NW(NWV, 2); }                        \
            else if (PU == 4) { PQ_LAUNCH_PARTIAL_NW(NWV, 4); }                        \
            else              { PQ_LAUNCH_PARTIAL_NW(NWV, 8); }                        \
        } while (0)
        if (PW == 4)       { PQ_DISPATCH_U(4); }
        else if (PW == 8)  { PQ_DISPATCH_U(8); }
        else if (PW == 16) { PQ_DISPATCH_U(16); }
        else               { PQ_DISPATCH_U(32); }
#undef PQ_DISPATCH_U
#undef PQ_LAUNCH_PARTIAL_NW
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (fr_on) return out;   // Reduce already done by the last CTA
    } else {
    pq_attend_partial_kernel<<<grid, kDMax, attend_smem, stream>>>(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(full_k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(full_v.data_ptr<at::Half>()),
        mask.data_ptr<float>(),
        topk_idx.data_ptr<int32_t>(),
        topk_val.data_ptr<float>(),
        ret_global.data_ptr<int64_t>(),
        reinterpret_cast<const __half*>(sb_k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(sb_v.data_ptr<at::Half>()),
        p_fmax.data_ptr<float>(), p_flse.data_ptr<float>(), p_facc.data_ptr<float>(),
        p_emax.data_ptr<float>(), p_else.data_ptr<float>(), p_eacc.data_ptr<float>(),
        p_omax.data_ptr<float>(), p_olse.data_ptr<float>(),
        D, N, K, F, static_cast<int>(cap),
        static_cast<float>(scale), md
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    dim3 g2(H, kG);
    int RW = 32;   // Default: 5.3x faster than the original single-warp reduce
    {
        const char* e = std::getenv("PQ_HSA_CUDA_REDWARPS");
        if (e) { RW = atoi(e); }
        if (RW < 1) RW = 1;
        if (RW > 32) RW = 32;
    }
    const float* p_bg_ptr = nullptr;
    if (bg_fused) {
        p_bg_ptr = ws.p_bg.data_ptr<float>();
        if (RW < 2) RW = 2;
    } else if (RS >= 1) {
        const int bg_smem = BW * kDMax * static_cast<int>(sizeof(float));
        dim3 gbg(H, kG, RS);
        if (BGMM) {
            if (!ws.mcorr.defined() || ws.mcorr.size(0) != H || ws.mcorr.size(2) != L
                || ws.mcorr.device() != q.device()) {
                ws.mcorr = torch::empty({H, kG, L}, opts_f);
            }
            dim3 gc(H, kG);
            pq_attend_corr_kernel<<<gc, 256, L * sizeof(float), stream>>>(
                topk_idx.data_ptr<int32_t>(), topk_val.data_ptr<float>(),
                list_ids.data_ptr<int32_t>(), list_mass.data_ptr<float>(),
                row_max.data_ptr<float>(), ws.mcorr.data_ptr<float>(),
                D, N, K, L);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            // smem for the bgmm CTA is BW*kG*128 floats; cap at 16 warps (32 KB) so we
            // stay under the 48 KB static limit without an opt-in attribute.
            // smem for the bgmm CTA is MW*kG*128 floats; cap it at 32 KB for any kG
            // and snap to the {4,8,16} warp instances the dispatch below has.
            int MW_CAP = (32 * 1024) / (kG * kDMax * static_cast<int>(sizeof(float)));
            MW_CAP = (MW_CAP >= 16) ? 16 : ((MW_CAP >= 8) ? 8 : 4);
            int MW = BW > MW_CAP ? MW_CAP : BW;
            if (MW > 16) MW = 16;
            // No opt-in attribute on this kernel -> stay under the 48 KiB default
            // dynamic-smem limit for large kG (16*8*128*4 = 64 KiB would fail to launch).
            while (MW > 4 && MW * kG * kDMax * static_cast<int>(sizeof(float)) > 48 * 1024) MW /= 2;
            const int mm_smem = MW * kG * kDMax * static_cast<int>(sizeof(float));
            dim3 gmm(H, RS);
#define PQ_LAUNCH_BGMM(BWV)                                                            \
            pq_attend_bgmm_kernel<BWV><<<gmm, BWV * 32, mm_smem, stream>>>(             \
                ws.mcorr.data_ptr<float>(),                                            \
                reinterpret_cast<const __half*>(centroids.data_ptr<at::Half>()),       \
                ws.p_bg.data_ptr<float>(), D, L, RS)
            if (MW == 4)       { PQ_LAUNCH_BGMM(4); }
            else if (MW == 8)  { PQ_LAUNCH_BGMM(8); }
            else               { PQ_LAUNCH_BGMM(16); }
#undef PQ_LAUNCH_BGMM
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            p_bg_ptr = ws.p_bg.data_ptr<float>();
            if (RW < 2) RW = 2;
            goto after_bg;
        }
#define PQ_LAUNCH_BG(BWV, BUV)                                                         \
        pq_attend_bg_kernel<BWV, BUV><<<gbg, BWV * 32, bg_smem, stream>>>(             \
            topk_idx.data_ptr<int32_t>(), topk_val.data_ptr<float>(),                  \
            list_ids.data_ptr<int32_t>(), list_mass.data_ptr<float>(),                 \
            reinterpret_cast<const __half*>(centroids.data_ptr<at::Half>()),           \
            row_max.data_ptr<float>(), ws.p_bg.data_ptr<float>(),                      \
            D, N, K, L, RS)
#define PQ_DISPATCH_BU(BWV)                                                            \
        do {                                                                           \
            if (BU == 1)      { PQ_LAUNCH_BG(BWV, 1); }                                \
            else if (BU == 2) { PQ_LAUNCH_BG(BWV, 2); }                                \
            else if (BU == 4) { PQ_LAUNCH_BG(BWV, 4); }                                \
            else              { PQ_LAUNCH_BG(BWV, 8); }                                \
        } while (0)
        if (BW == 4)       { PQ_DISPATCH_BU(4); }
        else if (BW == 8)  { PQ_DISPATCH_BU(8); }
        else if (BW == 16) { PQ_DISPATCH_BU(16); }
        else               { PQ_DISPATCH_BU(32); }
#undef PQ_DISPATCH_BU
#undef PQ_LAUNCH_BG
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        p_bg_ptr = ws.p_bg.data_ptr<float>();
        if (RW < 2) RW = 2;   // the p_bg path lives in the multi-warp reduce kernel
    }
after_bg:
    if (RW > 1) {
        size_t red_smem = (static_cast<size_t>(RW) * 128 + 128 + 128 + 8) * sizeof(float);
        pq_attend_reduce_mw_kernel<<<g2, RW * 32, red_smem, stream>>>(
            p_fmax.data_ptr<float>(), p_flse.data_ptr<float>(), p_facc.data_ptr<float>(),
            p_emax.data_ptr<float>(), p_else.data_ptr<float>(), p_eacc.data_ptr<float>(),
            p_omax.data_ptr<float>(), p_olse.data_ptr<float>(),
            topk_idx.data_ptr<int32_t>(),
            topk_val.data_ptr<float>(),
            list_ids.data_ptr<int32_t>(),
            list_mass.data_ptr<float>(),
            reinterpret_cast<const __half*>(centroids.data_ptr<at::Half>()),
            row_max.data_ptr<float>(),
            ret_exp.data_ptr<float>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            D, N, K, L, S, RW, p_bg_ptr, RS
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }
    pq_attend_reduce_kernel<<<g2, kDMax, 0, stream>>>(
        p_fmax.data_ptr<float>(), p_flse.data_ptr<float>(), p_facc.data_ptr<float>(),
        p_emax.data_ptr<float>(), p_else.data_ptr<float>(), p_eacc.data_ptr<float>(),
        p_omax.data_ptr<float>(), p_olse.data_ptr<float>(),
        topk_idx.data_ptr<int32_t>(),
        topk_val.data_ptr<float>(),
        list_ids.data_ptr<int32_t>(),
        list_mass.data_ptr<float>(),
        reinterpret_cast<const __half*>(centroids.data_ptr<at::Half>()),
        row_max.data_ptr<float>(),
        ret_exp.data_ptr<float>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        D, N, K, L, S
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


// ---------------------------------------------------------------------------
// Fused append-prep.  One launch replaces
//   (a) the Triton `_t83e_append_scatter` (shared-base + graph full/mask write)
//   (b) the eager `cg["q"].copy_(queries)` before graph replay.
// Pure data movement -> bit-identical with the kernels it replaces.
// ---------------------------------------------------------------------------
__global__ void pq_append_prep_kernel(
    const __half* __restrict__ nk,   // [H, D]  (contiguous)
    const __half* __restrict__ nv,   // [H, D]
    __half* __restrict__ shared_k,   // [H, cap, D]
    __half* __restrict__ shared_v,
    __half* __restrict__ full_k,     // [H, FC, D]
    __half* __restrict__ full_v,
    float*  __restrict__ maskp,      // [H, G, FC]
    const __half* __restrict__ q_src,// [HG, D] or null
    __half* __restrict__ q_dst,      // [HG, D] or null
    long long pos, long long spos,
    int H, int D, int cap, int FC, int G, int HG)
{
    const int b = blockIdx.x;
    const int t = threadIdx.x;
    if (b < H) {
        const int h = b;
        if (t < D) {
            const __half kv = nk[(long long)h * D + t];
            const __half vv = nv[(long long)h * D + t];
            shared_k[((long long)h * cap + pos) * D + t] = kv;
            shared_v[((long long)h * cap + pos) * D + t] = vv;
            full_k[((long long)h * FC + spos) * D + t] = kv;
            full_v[((long long)h * FC + spos) * D + t] = vv;
        }
        if (t < G) {
            maskp[((long long)h * G + t) * FC + spos] = 0.0f;
        }
    } else if (q_dst != nullptr) {
        const int r = b - H;
        if (r < HG && t < D) {
            q_dst[(long long)r * D + t] = q_src[(long long)r * D + t];
        }
    }
}

void pq_append_prep_cuda(
    torch::Tensor new_k, torch::Tensor new_v,
    torch::Tensor shared_k, torch::Tensor shared_v,
    torch::Tensor full_k, torch::Tensor full_v,
    torch::Tensor mask,
    c10::optional<torch::Tensor> q_src,
    c10::optional<torch::Tensor> q_dst,
    int64_t pos, int64_t spos)
{
    const int H  = (int)full_k.size(0);
    const int FC = (int)full_k.size(1);
    const int D  = (int)full_k.size(2);
    const int cap = (int)shared_k.size(1);
    const int G  = (int)mask.size(1);
    const __half* qs = nullptr;
    __half* qd = nullptr;
    int HG = 0;
    if (q_src.has_value() && q_dst.has_value()) {
        qs = reinterpret_cast<const __half*>(q_src->data_ptr<at::Half>());
        qd = reinterpret_cast<__half*>(q_dst->data_ptr<at::Half>());
        HG = (int)(q_dst->numel() / D);
    }
    const int blocks = H + HG;
    const int threads = (D > 128) ? 256 : 128;
    auto stream = at::cuda::getCurrentCUDAStream();
    pq_append_prep_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(new_k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(new_v.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(shared_k.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(shared_v.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(full_k.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(full_v.data_ptr<at::Half>()),
        mask.data_ptr<float>(),
        qs, qd,
        (long long)pos, (long long)spos,
        H, D, cap, FC, G, HG);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// ---------------------------------------------------------------------------
// Fused LUT prep.  ONE launch replaces the 5 kernels that currently
// stand between the query and the scan:
//   fp16 GEMM(lut)  +  scale  +  fp16 GEMM(list_scores)  +  scale  +  Triton
//   _h20_pair_build (pair LUT + list-score transpose).
// It writes the pair LUT [H,PAIRS,256,G] and list-score table [H,L,G] that the
// scan actually consumes; the intermediate [H,G,M,16] / [H,G,L] tensors are
// never materialised.  Arithmetic replicates the chain it replaces exactly:
//   half(dot_fp32) -> half(float(that) * scale) -> half(float(a)+float(b)).
// ---------------------------------------------------------------------------
template <int GG, int SDV>
__global__ void pq_lut_prep_kernel(
    const __half* __restrict__ q,      // [H, G, D]
    const __half* __restrict__ cb,     // [H, M, 16, SD]
    const __half* __restrict__ coarse, // [H, L, D]
    __half* __restrict__ pair_out,     // [H, PAIRS, 256, G]
    __half* __restrict__ list_out,     // [H, L, G]
    int H, int D, int M, int L, int PAIRS, int LPB, float scale)
{
    extern __shared__ float smem_lp[];
    const int h = blockIdx.x;
    const int job = blockIdx.y;
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;

    if (job == 0) {
        const int nlut = GG * M * 16;
        for (int e = tid; e < nlut; e += nthr) {
            const int g = e / (M * 16);
            const int rem = e - g * (M * 16);
            const int m = rem >> 4;
            const int c = rem & 15;
            const __half* qp = q + ((long long)h * GG + g) * D + m * SDV;
            const __half* cp = cb + (((long long)h * M + m) * 16 + c) * SDV;
            float acc = 0.f;
#pragma unroll
            for (int d = 0; d < SDV; ++d) acc += __half2float(qp[d]) * __half2float(cp[d]);
            // cublas fp16 GEMM rounds the fp32 accumulator to half; aten's scalar
            // mul then re-widens, scales and rounds again. Same two roundings.
            const __half h1 = __float2half(acc);
            smem_lp[e] = __half2float(__float2half(__half2float(h1) * scale));
        }
        __syncthreads();
        const int npair = PAIRS * 256 * GG;
        __half* pbase = pair_out + (long long)h * npair;
        for (int e = tid; e < npair; e += nthr) {
            const int g = static_cast<int>(static_cast<unsigned>(e) % static_cast<unsigned>(GG));
            const int t = e / GG;            // pair * 256 + byte
            const int byte = t & 255;
            const int pr = t >> 8;
            const float a = smem_lp[(g * M + 2 * pr) * 16 + (byte & 15)];
            const float b = smem_lp[(g * M + 2 * pr + 1) * 16 + (byte >> 4)];
            pbase[e] = __float2half(a + b);
        }
        return;
    }
    // ---- coarse list scores, transposed to [H, L, G] ------------------------
    for (int e = tid; e < GG * D; e += nthr) smem_lp[e] = __half2float(q[(long long)h * GG * D + e]);
    __syncthreads();
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = nthr >> 5;
    const int l0 = (job - 1) * LPB;
    for (int li = warp; li < LPB; li += nwarps) {
        const int l = l0 + li;
        if (l >= L) break;
        const __half* cp = coarse + (long long)h * L * D + (long long)l * D;
        float part[GG];
#pragma unroll
        for (int g = 0; g < GG; ++g) part[g] = 0.f;
        for (int d = lane; d < D; d += 32) {
            const float cv = __half2float(cp[d]);
#pragma unroll
            for (int g = 0; g < GG; ++g) part[g] += smem_lp[g * D + d] * cv;
        }
#pragma unroll
        for (int g = 0; g < GG; ++g) {
            float v = part[g];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
            if (lane == 0) {
                const __half h1 = __float2half(v);
                list_out[((long long)h * L + l) * GG + g] =
                    __float2half(__half2float(h1) * scale);
            }
        }
    }
}

std::vector<torch::Tensor> pq_lut_prep_cuda(
    torch::Tensor q,        // [H,G,D] fp16
    torch::Tensor cb,       // [H,M,16,SD] fp16
    torch::Tensor coarse,   // [H,L,D] fp16
    double scale)
{
    const int H = (int)q.size(0);
    const int G = (int)q.size(1);
    const int D = (int)q.size(2);
    const int M = (int)cb.size(1);
    const int SD = (int)cb.size(3);
    const int L = (int)coarse.size(1);
    const int PAIRS = M / 2;
    auto opts = q.options();
    auto pair_out = torch::empty({H, PAIRS, 256, G}, opts);
    auto list_out = torch::empty({H, L, G}, opts);
    int LPB = 8;   // lists per block: one warp per list, 8 warps/block
    { const char* e = std::getenv("PQ_HSA_LUTPREP_LPB"); if (e) { LPB = atoi(e); if (LPB < 1) LPB = 8; } }
    const int nchunk = (L + LPB - 1) / LPB;
    dim3 grid(H, 1 + nchunk);
    const int threads = 256;
    size_t smem = (size_t)std::max(G * M * 16, G * D) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(G == kG, "pq_lut_prep instance is compiled for G=", kG, ", got G=", G);
    TORCH_CHECK(SD == 16, "pq_lut_prep needs subdim=16");
    pq_lut_prep_kernel<kG, 16><<<grid, threads, smem, stream>>>(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(cb.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(coarse.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(pair_out.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(list_out.data_ptr<at::Half>()),
        H, D, M, L, PAIRS, LPB, (float)scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {pair_out, list_out};
}

// ---------------------------------------------------------------------------
// Block-stat fold.  ONE launch replaces the 5 aten kernels of
//   row_max = block_max.amax(-1)
//   exp_sum = (block_expsum * exp(block_max - row_max[...,None])).sum(-1)
// ---------------------------------------------------------------------------
__global__ void pq_block_reduce_kernel(
    const float* __restrict__ bmax,   // [R, NB]
    const float* __restrict__ bexp,   // [R, NB]
    float* __restrict__ row_max,      // [R]
    float* __restrict__ exp_sum,      // [R]
    int NB)
{
    extern __shared__ float sm_br[];
    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const float* bm = bmax + (long long)r * NB;
    const float* be = bexp + (long long)r * NB;
    float mx = -CUDART_INF_F;
    for (int i = tid; i < NB; i += nthr) mx = fmaxf(mx, bm[i]);
    sm_br[tid] = mx;
    __syncthreads();
    for (int s = nthr >> 1; s > 0; s >>= 1) {
        if (tid < s) sm_br[tid] = fmaxf(sm_br[tid], sm_br[tid + s]);
        __syncthreads();
    }
    const float rm = sm_br[0];
    __syncthreads();
    float acc = 0.f;
    for (int i = tid; i < NB; i += nthr) acc += be[i] * __expf(bm[i] - rm);
    sm_br[tid] = acc;
    __syncthreads();
    for (int s = nthr >> 1; s > 0; s >>= 1) {
        if (tid < s) sm_br[tid] += sm_br[tid + s];
        __syncthreads();
    }
    if (tid == 0) { row_max[r] = rm; exp_sum[r] = sm_br[0]; }
}

std::vector<torch::Tensor> pq_block_reduce_cuda(torch::Tensor bmax, torch::Tensor bexp) {
    TORCH_CHECK(bmax.dim() == 3 && bexp.dim() == 3, "block stats must be [H,G,NB]");
    const int H = (int)bmax.size(0), G = (int)bmax.size(1), NB = (int)bmax.size(2);
    const int R = H * G;
    auto row_max = torch::empty({H, G}, bmax.options());
    auto exp_sum = torch::empty({H, G}, bmax.options());
    int threads = 256;
    while (threads > 32 && threads > NB) threads >>= 1;
    auto stream = at::cuda::getCurrentCUDAStream();
    pq_block_reduce_kernel<<<R, threads, threads * sizeof(float), stream>>>(
        bmax.data_ptr<float>(), bexp.data_ptr<float>(),
        row_max.data_ptr<float>(), exp_sum.data_ptr<float>(), NB);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {row_max, exp_sum};
}

// ---------------------------------------------------------------------------
// One launch for the two dtype casts the attend epilogue needs
// (int64 top-k indices -> int32, fp16 top-k logits -> fp32).  Bit-identical.
// ---------------------------------------------------------------------------
__global__ void pq_cast_topk_kernel(
    const int64_t* __restrict__ idx64, const __half* __restrict__ val16,
    int32_t* __restrict__ idx32, float* __restrict__ val32, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        idx32[i] = (int32_t)idx64[i];
        val32[i] = __half2float(val16[i]);
    }
}

std::vector<torch::Tensor> pq_cast_topk_cuda(torch::Tensor idx64, torch::Tensor val16) {
    const long long n = idx64.numel();
    auto idx32 = torch::empty(idx64.sizes(), idx64.options().dtype(torch::kInt32));
    auto val32 = torch::empty(val16.sizes(), val16.options().dtype(torch::kFloat32));
    const int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 1024) blocks = 1024;
    if (blocks < 1) blocks = 1;
    auto stream = at::cuda::getCurrentCUDAStream();
    pq_cast_topk_kernel<<<blocks, threads, 0, stream>>>(
        idx64.data_ptr<int64_t>(),
        reinterpret_cast<const __half*>(val16.data_ptr<at::Half>()),
        idx32.data_ptr<int32_t>(), val32.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {idx32, val32};
}

// ---------------------------------------------------------------------------
// Host wrapper for pq_exact_attend_paged_kernel /
// pq_attend_partial_nw_paged_kernel. See the kernel-level comments above
// those kernels for the design. Includes the S>1 split-K path (dispatch
// mirrors pq_exact_attend_cuda's S>1 branch exactly -- same PARTWARPS/
// PARTUNROLL/BGSPLITS/BGWARPS/BGUNROLL/BGMM/REDWARPS env knobs, same
// reduce_mw/reduce/bg/bgmm/corr kernels, UNCHANGED) so that
// PQ_HSA_CUDA_SPLITS=8 (the adopted production value) is honoured by the
// paged path too.
// ---------------------------------------------------------------------------
torch::Tensor pq_exact_attend_paged_cuda(
    torch::Tensor q,
    torch::Tensor full_k,
    torch::Tensor full_v,
    torch::Tensor mask,
    torch::Tensor topk_idx,
    torch::Tensor topk_val,
    torch::Tensor ret_global,     // [H,N] int64, ABSOLUTE token positions
    torch::Tensor kv_cache,       // [2, num_blocks, block_size, num_kv_heads, D] fp16
    torch::Tensor block_table,    // [max_blocks] int32, this request's row
    torch::Tensor list_ids,
    torch::Tensor list_mass,
    torch::Tensor centroids,
    torch::Tensor row_max,
    torch::Tensor ret_exp,
    double scale,
    int64_t n_splits,
    int64_t mode
) {
    const int D = static_cast<int>(q.size(2));
    const int N = static_cast<int>(ret_global.size(1));
    const int K = static_cast<int>(topk_idx.size(2));
    const int F = static_cast<int>(full_k.size(1));
    const int L = static_cast<int>(centroids.size(1));
    const int H = static_cast<int>(q.size(0));
    // Accept both vLLM page layouts (see pq_paged_row). 5-D = previous path verbatim.
    int kv_layout = 0;
    int64_t num_blocks = 0;
    int block_size = 0;
    int num_kv_heads = 0;
    int64_t kv_sB = 0, kv_sH = 0, kv_sN = 0;   // 4-D only, units of D elements
    if (kv_cache.dim() == 5) {
        TORCH_CHECK(kv_cache.is_contiguous(), "kv_cache must be contiguous");
        TORCH_CHECK(kv_cache.size(0) == 2,
                    "kv_cache must be [2, num_blocks, block_size, num_kv_heads, head_dim]");
        num_blocks = kv_cache.size(1);
        block_size = static_cast<int>(kv_cache.size(2));
        num_kv_heads = static_cast<int>(kv_cache.size(3));
        TORCH_CHECK(static_cast<int64_t>(kv_cache.size(4)) == D,
                    "kv_cache head_dim must match q head_dim");
    } else {
        TORCH_CHECK(kv_cache.dim() == 4,
                    "kv_cache must be 5-D [2,B,bs,H,D] (vLLM<=0.8.5) or 4-D [B,H,bs,2D] (vLLM>=0.10)");
        kv_layout = 1;
        num_blocks = kv_cache.size(0);
        num_kv_heads = static_cast<int>(kv_cache.size(1));
        block_size = static_cast<int>(kv_cache.size(2));
        TORCH_CHECK(static_cast<int64_t>(kv_cache.size(3)) == 2 * static_cast<int64_t>(D),
                    "4-D kv_cache last dim must be 2*head_dim (K|V concatenated)");
        TORCH_CHECK(kv_cache.stride(3) == 1, "4-D kv_cache: K|V dim must be innermost (stride 1)");
        TORCH_CHECK(kv_cache.stride(0) % D == 0 && kv_cache.stride(1) % D == 0 && kv_cache.stride(2) % D == 0,
                    "4-D kv_cache: block/head/token strides must be multiples of head_dim");
        kv_sB = kv_cache.stride(0) / D;
        kv_sH = kv_cache.stride(1) / D;
        kv_sN = kv_cache.stride(2) / D;
    }
    auto out = torch::empty({H, kG, D}, q.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int attend_smem = (8 + 4 * kDMax) * sizeof(float);
    int md = static_cast<int>(mode);

    const __half* kv_ptr = reinterpret_cast<const __half*>(kv_cache.data_ptr<at::Half>());
    const __half* k_base = kv_ptr;
    const __half* v_base = (kv_layout == 0)
        ? kv_ptr + num_blocks * static_cast<int64_t>(block_size) * num_kv_heads * D
        : kv_ptr + D;   // 4-D page, V half sits D elements after K in every slot
    const int32_t* bt_ptr = block_table.data_ptr<int32_t>();

    int S = static_cast<int>(n_splits);
    if (S < 1) S = 1;
    if (S > 32) S = 32;

    if (S <= 1) {
        dim3 grid(H, kG);
        pq_exact_attend_paged_kernel<<<grid, kDMax, attend_smem, stream>>>(
            reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(full_k.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(full_v.data_ptr<at::Half>()),
            mask.data_ptr<float>(),
            topk_idx.data_ptr<int32_t>(),
            topk_val.data_ptr<float>(),
            ret_global.data_ptr<int64_t>(),
            k_base,
            v_base,
            bt_ptr,
            list_ids.data_ptr<int32_t>(),
            list_mass.data_ptr<float>(),
            reinterpret_cast<const __half*>(centroids.data_ptr<at::Half>()),
            row_max.data_ptr<float>(),
            ret_exp.data_ptr<float>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            D, N, K, F, L, block_size, num_kv_heads,
            static_cast<float>(scale), md, kv_layout, kv_sB, kv_sH, kv_sN
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }

    // --- S>1: split-K partial (paged) + UNCHANGED bg/reduce kernels ---------
    struct WSP {
        int H = 0, S = 0, D = 0;
        torch::Tensor p_fmax, p_flse, p_facc, p_emax, p_else, p_eacc, p_omax, p_olse;
        int RS = 0;
        torch::Tensor p_bg;
        torch::Tensor mcorr;
        torch::Tensor sem;
    };
    static WSP ws;
    auto opts_f = q.options().dtype(torch::kFloat32);
    if (ws.H != H || ws.S != S || ws.D != D || !ws.p_fmax.defined()
        || ws.p_fmax.device() != q.device()) {
        ws.H = H; ws.S = S; ws.D = D;
        ws.p_fmax = torch::empty({H, kG, S}, opts_f);
        ws.p_flse = torch::empty({H, kG, S}, opts_f);
        ws.p_facc = torch::empty({H, kG, S, D}, opts_f);
        ws.p_emax = torch::empty({H, kG, S}, opts_f);
        ws.p_else = torch::empty({H, kG, S}, opts_f);
        ws.p_eacc = torch::empty({H, kG, S, D}, opts_f);
        ws.p_omax = torch::empty({H, kG, S}, opts_f);
        ws.p_olse = torch::empty({H, kG, S}, opts_f);
    }
    auto& p_fmax = ws.p_fmax; auto& p_flse = ws.p_flse; auto& p_facc = ws.p_facc;
    auto& p_emax = ws.p_emax; auto& p_else = ws.p_else; auto& p_eacc = ws.p_eacc;
    auto& p_omax = ws.p_omax; auto& p_olse = ws.p_olse;

    dim3 grid(H, kG, S);
    int RS = 0, BW = 8, BU = 1, BGMM = 1, BGFUSE = 0;
    {
        const char* e = std::getenv("PQ_HSA_CUDA_BGSPLITS");
        if (e) { RS = atoi(e); }
        if (RS < 0) RS = 0;
        if (RS > 32) RS = 32;
        const char* e2 = std::getenv("PQ_HSA_CUDA_BGWARPS");
        if (e2) { BW = atoi(e2); }
        if (BW != 4 && BW != 8 && BW != 16 && BW != 32) BW = 8;
        const char* e3 = std::getenv("PQ_HSA_CUDA_BGUNROLL");
        if (e3) { BU = atoi(e3); }
        if (BU != 1 && BU != 2 && BU != 4 && BU != 8) BU = 1;
        const char* e4 = std::getenv("PQ_HSA_CUDA_BGMM");
        if (e4) { BGMM = atoi(e4); }
        const char* e5 = std::getenv("PQ_HSA_CUDA_BGFUSE");
        if (e5) { BGFUSE = atoi(e5); }
    }
    if (BGFUSE && RS >= 1) { RS = S; BGMM = 0; }
    if (RS >= 1) {
        if (ws.RS != RS || !ws.p_bg.defined() || ws.p_bg.size(0) != H
            || ws.p_bg.size(3) != D || ws.p_bg.device() != q.device()) {
            ws.RS = RS;
            ws.p_bg = torch::empty({H, kG, RS, D}, opts_f);
        }
    }
    { const char* e = std::getenv("PQ_HSA_REDUCE_FUSED"); if (e && atoi(e) && S > 1) { BGFUSE = 1; RS = S; BGMM = 0;
        if (ws.RS != RS || !ws.p_bg.defined() || ws.p_bg.size(0) != H || ws.p_bg.size(3) != D || ws.p_bg.device() != q.device()) { ws.RS = RS; ws.p_bg = torch::empty({H, kG, RS, D}, opts_f); } } }
    const bool bg_fused = (BGFUSE && RS >= 1);
    int PW = 4;
    {
        const char* e = std::getenv("PQ_HSA_CUDA_PARTWARPS");
        if (e) { PW = atoi(e); }
        if (PW != 4 && PW != 8 && PW != 16 && PW != 32) PW = 4;
    }
    int PU = 1;
    {
        const char* e = std::getenv("PQ_HSA_CUDA_PARTUNROLL");
        if (e) { PU = atoi(e); }
        if (PU != 1 && PU != 2 && PU != 4 && PU != 8) PU = 1;
    }
    int MASKSKIP = 0;
    { const char* e = std::getenv("PQ_HSA_MASKSKIP"); if (e) MASKSKIP = atoi(e); }
    // Fold the split reduce into the partial kernel (last CTA per head).  Needs the
    // background bracket inside the partial too (bg fused, RS=S), so force that layout.
    int FR = 0;
    { const char* e = std::getenv("PQ_HSA_REDUCE_FUSED"); if (e) FR = atoi(e); }
    if (FR && S > 1) {
        RS = S; BGMM = 0;
        if (ws.RS != RS || !ws.p_bg.defined() || ws.p_bg.size(0) != H
            || ws.p_bg.size(3) != D || ws.p_bg.device() != q.device()) {
            ws.RS = RS;
            ws.p_bg = torch::empty({H, kG, RS, D}, opts_f);
        }
        if (!ws.sem.defined() || ws.sem.size(0) != H || ws.sem.device() != q.device()) {
            ws.sem = torch::zeros({H}, q.options().dtype(torch::kInt32));
        }
    } else { FR = 0; }
    const bool fr_on = (FR != 0);
    int* fr_sem_ptr = fr_on ? ws.sem.data_ptr<int>() : nullptr;
    const float* fr_ret_exp_ptr = fr_on ? ret_exp.data_ptr<float>() : nullptr;
    __half* fr_out_ptr = fr_on ? reinterpret_cast<__half*>(out.data_ptr<at::Half>()) : nullptr;

    const int part_smem = (2 * PW + PW * kDMax) * sizeof(float);
#define PQ_LAUNCH_PARTIAL_NW_PAGED(NWV, UV)                                            \
    pq_attend_partial_nw_paged_kernel<NWV, UV><<<grid, NWV * 32, part_smem, stream>>>( \
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),                       \
        reinterpret_cast<const __half*>(full_k.data_ptr<at::Half>()),                  \
        reinterpret_cast<const __half*>(full_v.data_ptr<at::Half>()),                  \
        mask.data_ptr<float>(),                                                        \
        topk_idx.data_ptr<int32_t>(),                                                  \
        topk_val.data_ptr<float>(),                                                    \
        ret_global.data_ptr<int64_t>(),                                                \
        k_base, v_base, bt_ptr,                                                        \
        p_fmax.data_ptr<float>(), p_flse.data_ptr<float>(), p_facc.data_ptr<float>(),   \
        p_emax.data_ptr<float>(), p_else.data_ptr<float>(), p_eacc.data_ptr<float>(),   \
        p_omax.data_ptr<float>(), p_olse.data_ptr<float>(),                            \
        D, N, K, F, block_size, num_kv_heads, static_cast<float>(scale), md,            \
        list_ids.data_ptr<int32_t>(), list_mass.data_ptr<float>(),                     \
        reinterpret_cast<const __half*>(centroids.data_ptr<at::Half>()),               \
        row_max.data_ptr<float>(),                                                     \
        bg_fused ? ws.p_bg.data_ptr<float>() : nullptr, L, MASKSKIP,                     \
        fr_sem_ptr, fr_ret_exp_ptr, fr_out_ptr, kv_layout, kv_sB, kv_sH, kv_sN)
#define PQ_DISPATCH_U_PAGED(NWV)                                                       \
    do {                                                                               \
        if (PU == 1)      { PQ_LAUNCH_PARTIAL_NW_PAGED(NWV, 1); }                      \
        else if (PU == 2) { PQ_LAUNCH_PARTIAL_NW_PAGED(NWV, 2); }                      \
        else if (PU == 4) { PQ_LAUNCH_PARTIAL_NW_PAGED(NWV, 4); }                      \
        else              { PQ_LAUNCH_PARTIAL_NW_PAGED(NWV, 8); }                      \
    } while (0)
    if (PW == 4)       { PQ_DISPATCH_U_PAGED(4); }
    else if (PW == 8)  { PQ_DISPATCH_U_PAGED(8); }
    else if (PW == 16) { PQ_DISPATCH_U_PAGED(16); }
    else               { PQ_DISPATCH_U_PAGED(32); }
#undef PQ_DISPATCH_U_PAGED
#undef PQ_LAUNCH_PARTIAL_NW_PAGED
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (fr_on) return out;

    dim3 g2(H, kG);
    int RW = 32;
    {
        const char* e = std::getenv("PQ_HSA_CUDA_REDWARPS");
        if (e) { RW = atoi(e); }
        if (RW < 1) RW = 1;
        if (RW > 32) RW = 32;
    }
    const float* p_bg_ptr = nullptr;
    if (bg_fused) {
        p_bg_ptr = ws.p_bg.data_ptr<float>();
        if (RW < 2) RW = 2;
    } else if (RS >= 1) {
        const int bg_smem = BW * kDMax * static_cast<int>(sizeof(float));
        dim3 gbg(H, kG, RS);
        if (BGMM) {
            if (!ws.mcorr.defined() || ws.mcorr.size(0) != H || ws.mcorr.size(2) != L
                || ws.mcorr.device() != q.device()) {
                ws.mcorr = torch::empty({H, kG, L}, opts_f);
            }
            dim3 gc(H, kG);
            pq_attend_corr_kernel<<<gc, 256, L * sizeof(float), stream>>>(
                topk_idx.data_ptr<int32_t>(), topk_val.data_ptr<float>(),
                list_ids.data_ptr<int32_t>(), list_mass.data_ptr<float>(),
                row_max.data_ptr<float>(), ws.mcorr.data_ptr<float>(),
                D, N, K, L);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            int MW_CAP = (32 * 1024) / (kG * kDMax * static_cast<int>(sizeof(float)));
            MW_CAP = (MW_CAP >= 16) ? 16 : ((MW_CAP >= 8) ? 8 : 4);
            int MW = BW > MW_CAP ? MW_CAP : BW;
            if (MW > 16) MW = 16;
            // No opt-in attribute on this kernel -> stay under the 48 KiB default
            // dynamic-smem limit for large kG (16*8*128*4 = 64 KiB would fail to launch).
            while (MW > 4 && MW * kG * kDMax * static_cast<int>(sizeof(float)) > 48 * 1024) MW /= 2;
            const int mm_smem = MW * kG * kDMax * static_cast<int>(sizeof(float));
            dim3 gmm(H, RS);
#define PQ_LAUNCH_BGMM_P(BWV)                                                          \
            pq_attend_bgmm_kernel<BWV><<<gmm, BWV * 32, mm_smem, stream>>>(             \
                ws.mcorr.data_ptr<float>(),                                            \
                reinterpret_cast<const __half*>(centroids.data_ptr<at::Half>()),       \
                ws.p_bg.data_ptr<float>(), D, L, RS)
            if (MW == 4)       { PQ_LAUNCH_BGMM_P(4); }
            else if (MW == 8)  { PQ_LAUNCH_BGMM_P(8); }
            else               { PQ_LAUNCH_BGMM_P(16); }
#undef PQ_LAUNCH_BGMM_P
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            p_bg_ptr = ws.p_bg.data_ptr<float>();
            if (RW < 2) RW = 2;
            goto after_bg_p;
        }
#define PQ_LAUNCH_BG_P(BWV, BUV)                                                       \
        pq_attend_bg_kernel<BWV, BUV><<<gbg, BWV * 32, bg_smem, stream>>>(              \
            topk_idx.data_ptr<int32_t>(), topk_val.data_ptr<float>(),                  \
            list_ids.data_ptr<int32_t>(), list_mass.data_ptr<float>(),                 \
            reinterpret_cast<const __half*>(centroids.data_ptr<at::Half>()),           \
            row_max.data_ptr<float>(), ws.p_bg.data_ptr<float>(),                      \
            D, N, K, L, RS)
#define PQ_DISPATCH_BU_P(BWV)                                                          \
        do {                                                                           \
            if (BU == 1)      { PQ_LAUNCH_BG_P(BWV, 1); }                              \
            else if (BU == 2) { PQ_LAUNCH_BG_P(BWV, 2); }                              \
            else if (BU == 4) { PQ_LAUNCH_BG_P(BWV, 4); }                              \
            else              { PQ_LAUNCH_BG_P(BWV, 8); }                              \
        } while (0)
        if (BW == 4)       { PQ_DISPATCH_BU_P(4); }
        else if (BW == 8)  { PQ_DISPATCH_BU_P(8); }
        else if (BW == 16) { PQ_DISPATCH_BU_P(16); }
        else               { PQ_DISPATCH_BU_P(32); }
#undef PQ_DISPATCH_BU_P
#undef PQ_LAUNCH_BG_P
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        p_bg_ptr = ws.p_bg.data_ptr<float>();
        if (RW < 2) RW = 2;
    }
after_bg_p:
    if (RW > 1) {
        size_t red_smem = (static_cast<size_t>(RW) * 128 + 128 + 128 + 8) * sizeof(float);
        pq_attend_reduce_mw_kernel<<<g2, RW * 32, red_smem, stream>>>(
            p_fmax.data_ptr<float>(), p_flse.data_ptr<float>(), p_facc.data_ptr<float>(),
            p_emax.data_ptr<float>(), p_else.data_ptr<float>(), p_eacc.data_ptr<float>(),
            p_omax.data_ptr<float>(), p_olse.data_ptr<float>(),
            topk_idx.data_ptr<int32_t>(),
            topk_val.data_ptr<float>(),
            list_ids.data_ptr<int32_t>(),
            list_mass.data_ptr<float>(),
            reinterpret_cast<const __half*>(centroids.data_ptr<at::Half>()),
            row_max.data_ptr<float>(),
            ret_exp.data_ptr<float>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            D, N, K, L, S, RW, p_bg_ptr, RS
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }
    pq_attend_reduce_kernel<<<g2, kDMax, 0, stream>>>(
        p_fmax.data_ptr<float>(), p_flse.data_ptr<float>(), p_facc.data_ptr<float>(),
        p_emax.data_ptr<float>(), p_else.data_ptr<float>(), p_eacc.data_ptr<float>(),
        p_omax.data_ptr<float>(), p_olse.data_ptr<float>(),
        topk_idx.data_ptr<int32_t>(),
        topk_val.data_ptr<float>(),
        list_ids.data_ptr<int32_t>(),
        list_mass.data_ptr<float>(),
        reinterpret_cast<const __half*>(centroids.data_ptr<at::Half>()),
        row_max.data_ptr<float>(),
        ret_exp.data_ptr<float>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        D, N, K, L, S
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


// ---------------------------------------------------------------------------
// (opt-in, PQ_HSA_TOPK_RADIX=1): exact top-k over fp16 rows in ONE kernel.
// One CTA per row; two 8-bit radix passes on the fp16 sort key (smem histograms,
// per-warp private copies) find the k-th largest key, a third pass emits the
// indices/values of all keys above it plus the first (k - above) ties.  The
// selected SET equals torch.topk(..., sorted=False) up to tie order.  Replaces the
// aten mbtopk pipeline (8 launches, ~30 us fixed cost) for the small row counts
// (H*G <= 64) that PQ-HSA decode produces.
// ---------------------------------------------------------------------------
namespace e33 {
constexpr int kNT = 1024;
constexpr int kNWARP = kNT / 32;

__device__ __forceinline__ uint16_t key16(__half h) {
    uint16_t bits = __half_as_ushort(h);
    return (bits & 0x8000u) ? static_cast<uint16_t>((~bits) & 0xFFFFu)
                            : static_cast<uint16_t>(bits ^ 0x8000u);
}

// 256-bin histogram of byte `sel(key)` over the row, restricted to keys whose high
// byte equals `hi_req` when `pass2` is set.  Per-warp private smem copies.
template <bool PASS2>
__device__ __forceinline__ void hist_pass(const __half* __restrict__ row, int N, int hi_req,
                                          int* __restrict__ hw /* [kNWARP][256] */)
{
    const int wid = threadIdx.x >> 5;
    int* my = hw + wid * 256;
    for (int b = threadIdx.x; b < kNWARP * 256; b += kNT) hw[b] = 0;
    __syncthreads();
    // unaligned head + tail handled scalar; middle via 16-byte loads (8 halfs)
    const uintptr_t addr = reinterpret_cast<uintptr_t>(row);
    int head = static_cast<int>((16 - (addr & 15)) & 15) / 2;
    if (head > N) head = N;
    const int n_mid = (N - head) / 8;
    const int mid_end = head + n_mid * 8;
    for (int i = threadIdx.x; i < head; i += kNT) {
        const uint16_t k = key16(row[i]);
        const int b = PASS2 ? ((k >> 8) == hi_req ? (k & 255) : -1) : (k >> 8);
        if (b >= 0) atomicAdd(&my[b], 1);
    }
    const uint4* mid = reinterpret_cast<const uint4*>(row + head);
#pragma unroll 2
    for (int v = threadIdx.x; v < n_mid; v += kNT) {
        const uint4 w = __ldg(mid + v);
        const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const uint16_t bits = static_cast<uint16_t>((ws[j] >> (16 * hh)) & 0xFFFFu);
                const uint16_t k = (bits & 0x8000u) ? static_cast<uint16_t>((~bits) & 0xFFFFu)
                                                    : static_cast<uint16_t>(bits ^ 0x8000u);
                const int b = PASS2 ? ((k >> 8) == hi_req ? (k & 255) : -1) : (k >> 8);
                if (b >= 0) atomicAdd(&my[b], 1);
            }
        }
    }
    for (int i = mid_end + threadIdx.x; i < N; i += kNT) {
        const uint16_t k = key16(row[i]);
        const int b = PASS2 ? ((k >> 8) == hi_req ? (k & 255) : -1) : (k >> 8);
        if (b >= 0) atomicAdd(&my[b], 1);
    }
    __syncthreads();
}

// Reduce per-warp histograms into h[256], build "count strictly above bin b" (cg[b]),
// and pick the bin where cg[b] < need <= cg[b] + h[b].  Returns via smem out[0]=bin,
// out[1]=cg[bin].  Runs on the first 256 threads.
__device__ __forceinline__ void pick_bin(const int* __restrict__ hw, int need,
                                         int* __restrict__ h, int* __restrict__ cg, int* __restrict__ out)
{
    const int t = threadIdx.x;
    if (t < 256) {
        int acc = 0;
#pragma unroll 4
        for (int w = 0; w < kNWARP; ++w) acc += hw[w * 256 + t];
        h[t] = acc;
    }
    __syncthreads();
    // exclusive suffix sum: cg[b] = sum_{b' > b} h[b']; do a Hillis-Steele inclusive scan
    // over reversed index r = 255 - b, then cg[b] = incl[r] - h[b].
    if (t < 256) cg[t] = h[255 - t];          // cg temporarily holds reversed h
    __syncthreads();
    for (int off = 1; off < 256; off <<= 1) {
        int v = 0;
        if (t < 256 && t >= off) v = cg[t - off];
        __syncthreads();
        if (t < 256) cg[t] += v;
        __syncthreads();
    }
    // now cg[r] = inclusive sum of reversed h up to r  => count(keys with bin >= 255-r)
    if (t < 256) {
        const int b = 255 - t;
        const int ge = cg[t];              // count with bin >= b
        const int gt = ge - h[b];          // count with bin >  b
        if (gt < need && need <= ge) { out[0] = b; out[1] = gt; }
    }
    __syncthreads();
}

__global__ void __launch_bounds__(kNT) pq_topk_radix_kernel(
    const __half* __restrict__ scores,  // [R, N]
    int N, int K,
    int32_t* __restrict__ out_idx,      // [R, K]
    float* __restrict__ out_val)        // [R, K]
{
    __shared__ int hw[kNWARP * 256];
    __shared__ int hsum[256];
    __shared__ int cgs[256];
    __shared__ int pick[2];
    __shared__ int counters[2];         // [0] above-threshold slots, [1] tie slots
    const int row = blockIdx.x;
    const __half* __restrict__ r = scores + static_cast<long long>(row) * N;
    int32_t* __restrict__ oi = out_idx + static_cast<long long>(row) * K;
    float* __restrict__ ov = out_val + static_cast<long long>(row) * K;
    if (K <= 0) return;
    if (K >= N) {   // degenerate: take everything (matches torch.topk(k=N))
        for (int i = threadIdx.x; i < N; i += kNT) { oi[i] = i; ov[i] = __half2float(r[i]); }
        return;
    }
    // pass 1: high byte
    hist_pass<false>(r, N, 0, hw);
    pick_bin(hw, K, hsum, cgs, pick);
    const int b1 = pick[0];
    const int above1 = pick[1];
    __syncthreads();
    // pass 2: low byte among keys with high byte == b1
    hist_pass<true>(r, N, b1, hw);
    pick_bin(hw, K - above1, hsum, cgs, pick);
    const int b2 = pick[0];
    const int above2 = pick[1];
    const uint16_t thr = static_cast<uint16_t>((b1 << 8) | b2);
    const int n_above = above1 + above2;        // keys strictly greater than thr
    const int n_tie = K - n_above;              // >= 1 ties to take
    if (threadIdx.x == 0) { counters[0] = 0; counters[1] = 0; }
    __syncthreads();
    // pass 3: emit.  Warp-aggregated slot allocation (one smem atomic per warp per chunk).
    const int lane = threadIdx.x & 31;
    for (int base = 0; base < N; base += kNT) {
        const int i = base + threadIdx.x;
        int cls = 0;   // 0 none, 1 above, 2 tie
        __half hv;
        if (i < N) {
            hv = r[i];
            const uint16_t k = key16(hv);
            cls = (k > thr) ? 1 : ((k == thr) ? 2 : 0);
        }
        const unsigned m_above = __ballot_sync(0xffffffffu, cls == 1);
        const unsigned m_tie = __ballot_sync(0xffffffffu, cls == 2);
        int base_above = 0, base_tie = 0;
        if (lane == 0) {
            if (m_above) base_above = atomicAdd(&counters[0], __popc(m_above));
            if (m_tie) base_tie = atomicAdd(&counters[1], __popc(m_tie));
        }
        base_above = __shfl_sync(0xffffffffu, base_above, 0);
        base_tie = __shfl_sync(0xffffffffu, base_tie, 0);
        if (cls == 1) {
            const int slot = base_above + __popc(m_above & ((1u << lane) - 1u));
            if (slot < n_above) { oi[slot] = i; ov[slot] = __half2float(hv); }
        } else if (cls == 2) {
            const int t = base_tie + __popc(m_tie & ((1u << lane) - 1u));
            if (t < n_tie) { oi[n_above + t] = i; ov[n_above + t] = __half2float(hv); }
        }
    }
}
}  // namespace e33

std::vector<torch::Tensor> pq_topk_radix_cuda(torch::Tensor scores, int64_t k) {
    TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == torch::kFloat16, "pq_topk_radix: fp16 CUDA scores");
    TORCH_CHECK(scores.dim() >= 1);
    auto sc = scores.contiguous();
    const int N = static_cast<int>(sc.size(-1));
    const long long R = sc.numel() / std::max(1, N);
    int K = static_cast<int>(k);
    if (K > N) K = N;
    TORCH_CHECK(K >= 1, "pq_topk_radix: k must be >= 1");
    std::vector<int64_t> oshape(sc.sizes().begin(), sc.sizes().end());
    oshape.back() = K;
    auto out_idx = torch::empty(oshape, sc.options().dtype(torch::kInt32));
    auto out_val = torch::empty(oshape, sc.options().dtype(torch::kFloat32));
    auto stream = at::cuda::getCurrentCUDAStream();
    e33::pq_topk_radix_kernel<<<static_cast<unsigned>(R), e33::kNT, 0, stream>>>(
        reinterpret_cast<const __half*>(sc.data_ptr<at::Half>()), N, K,
        out_idx.data_ptr<int32_t>(), out_val.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out_idx, out_val};
}

// ---------------------------------------------------------------------------
// (GSR fragment cleanup): kG=1 instance of pq_lut_prep for the group-mean query
// (group-shared retrieval).  Same kernel template, GG=1, no G==kG requirement.
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> pq_lut_prep_g1_cuda(
    torch::Tensor q,        // [H,1,D] fp16
    torch::Tensor cb,       // [H,M,16,SD] fp16
    torch::Tensor coarse,   // [H,L,D] fp16
    double scale)
{
    const int H = (int)q.size(0);
    const int G = (int)q.size(1);
    const int D = (int)q.size(2);
    const int M = (int)cb.size(1);
    const int SD = (int)cb.size(3);
    const int L = (int)coarse.size(1);
    const int PAIRS = M / 2;
    TORCH_CHECK(G == 1, "pq_lut_prep_g1 expects G=1, got ", G);
    TORCH_CHECK(SD == 16, "pq_lut_prep_g1 needs subdim=16");
    auto opts = q.options();
    auto pair_out = torch::empty({H, PAIRS, 256, 1}, opts);
    auto list_out = torch::empty({H, L, 1}, opts);
    int LPB = 8;
    { const char* e = std::getenv("PQ_HSA_LUTPREP_LPB"); if (e) { LPB = atoi(e); if (LPB < 1) LPB = 8; } }
    const int nchunk = (L + LPB - 1) / LPB;
    dim3 grid(H, 1 + nchunk);
    const int threads = 256;
    size_t smem = (size_t)std::max(1 * M * 16, 1 * D) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    pq_lut_prep_kernel<1, 16><<<grid, threads, smem, stream>>>(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(cb.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(coarse.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(pair_out.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(list_out.data_ptr<at::Half>()),
        H, D, M, L, PAIRS, LPB, (float)scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {pair_out, list_out};
}

// ---------------------------------------------------------------------------
// (GSR fragment cleanup): ONE launch broadcasts the per-KV-head retrieval result
// (idx [H,1,K], val [H,1,K], row_max [H,1], exp_sum [H,1], mass [H,1,L]) to the
// kG query heads, emitting exactly the int32 / fp32 buffers the attend kernels
// consume (replaces 5 expand().contiguous() copies + the 2 dtype casts).
// ---------------------------------------------------------------------------
template <bool IDX64, bool VAL16>
__global__ void pq_gsr_expand_kernel(
    const void* __restrict__ idx_in, const void* __restrict__ val_in,
    const float* __restrict__ rmax_in, const float* __restrict__ rexp_in,
    const float* __restrict__ mass_in,
    int32_t* __restrict__ idx_out, float* __restrict__ val_out,
    float* __restrict__ rmax_out, float* __restrict__ rexp_out, float* __restrict__ mass_out,
    int K, int L)
{
    const int h = blockIdx.x;
    const int g = blockIdx.y;
    const long long src = static_cast<long long>(h) * K;
    const long long dst = static_cast<long long>(h * kG + g) * K;
    for (int t = threadIdx.x; t < K; t += blockDim.x) {
        int32_t iv = IDX64 ? static_cast<int32_t>(reinterpret_cast<const int64_t*>(idx_in)[src + t])
                           : reinterpret_cast<const int32_t*>(idx_in)[src + t];
        float vv = VAL16 ? __half2float(reinterpret_cast<const __half*>(val_in)[src + t])
                         : reinterpret_cast<const float*>(val_in)[src + t];
        idx_out[dst + t] = iv;
        val_out[dst + t] = vv;
    }
    const long long msrc = static_cast<long long>(h) * L;
    const long long mdst = static_cast<long long>(h * kG + g) * L;
    for (int l = threadIdx.x; l < L; l += blockDim.x) mass_out[mdst + l] = mass_in[msrc + l];
    if (threadIdx.x == 0) { rmax_out[h * kG + g] = rmax_in[h]; rexp_out[h * kG + g] = rexp_in[h]; }
}

std::vector<torch::Tensor> pq_gsr_expand_cuda(
    torch::Tensor idx, torch::Tensor val, torch::Tensor rmax, torch::Tensor rexp, torch::Tensor mass)
{
    const int H = (int)idx.size(0);
    TORCH_CHECK(idx.size(1) == 1 && val.size(1) == 1 && mass.size(1) == 1, "pq_gsr_expand expects Gr=1 inputs");
    const int K = (int)idx.size(2);
    const int L = (int)mass.size(2);
    struct WSE { int H = 0, K = 0, L = 0; torch::Tensor i, v, m, x, mm; };
    static WSE ws;
    if (ws.H != H || ws.K != K || ws.L != L || !ws.i.defined() || ws.i.device() != idx.device()) {
        ws.H = H; ws.K = K; ws.L = L;
        auto of = mass.options().dtype(torch::kFloat32);
        ws.i = torch::empty({H, kG, K}, idx.options().dtype(torch::kInt32));
        ws.v = torch::empty({H, kG, K}, of);
        ws.m = torch::empty({H, kG}, of);
        ws.x = torch::empty({H, kG}, of);
        ws.mm = torch::empty({H, kG, L}, of);
    }
    auto ic = idx.contiguous(); auto vc = val.contiguous();
    auto rm = rmax.to(torch::kFloat32).contiguous(); auto re = rexp.to(torch::kFloat32).contiguous();
    auto mc = mass.to(torch::kFloat32).contiguous();
    const bool i64 = (ic.scalar_type() == torch::kInt64);
    const bool v16 = (vc.scalar_type() == torch::kFloat16);
    TORCH_CHECK(i64 || ic.scalar_type() == torch::kInt32, "idx must be int32/int64");
    TORCH_CHECK(v16 || vc.scalar_type() == torch::kFloat32, "val must be fp16/fp32");
    dim3 grid(H, kG);
    auto stream = at::cuda::getCurrentCUDAStream();
#define PQ_GSR_LAUNCH(A, B)                                                            \
    pq_gsr_expand_kernel<A, B><<<grid, 256, 0, stream>>>(                              \
        ic.data_ptr(), vc.data_ptr(), rm.data_ptr<float>(), re.data_ptr<float>(),      \
        mc.data_ptr<float>(), ws.i.data_ptr<int32_t>(), ws.v.data_ptr<float>(),        \
        ws.m.data_ptr<float>(), ws.x.data_ptr<float>(), ws.mm.data_ptr<float>(), K, L)
    if (i64 && v16) { PQ_GSR_LAUNCH(true, true); }
    else if (i64)   { PQ_GSR_LAUNCH(true, false); }
    else if (v16)   { PQ_GSR_LAUNCH(false, true); }
    else            { PQ_GSR_LAUNCH(false, false); }
#undef PQ_GSR_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {ws.i, ws.v, ws.m, ws.x, ws.mm};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pq_scan_select", &pq_scan_select_cuda);
    m.def("pq_scan_only", &pq_scan_only_cuda);
    m.def("pq_merge_only", &pq_merge_only_cuda);
    m.def("pq_exact_attend", &pq_exact_attend_cuda);
    m.def("pq_append_prep", &pq_append_prep_cuda);
    m.def("pq_lut_prep", &pq_lut_prep_cuda);
    m.def("pq_block_reduce", &pq_block_reduce_cuda);
    m.def("pq_cast_topk", &pq_cast_topk_cuda);
    m.def("pq_exact_attend_paged", &pq_exact_attend_paged_cuda);
    m.def("pq_topk_radix", &pq_topk_radix_cuda);
    m.def("pq_lut_prep_g1", &pq_lut_prep_g1_cuda);
    m.def("pq_gsr_expand", &pq_gsr_expand_cuda);
}
