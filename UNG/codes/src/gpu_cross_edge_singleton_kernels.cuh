// Singleton-target fastpath kernels.
//
// These kernels handle target groups with nx == 1. They are intentionally
// separated from the larger fused group kernels because their launch condition
// and output semantics are special: either write local top1 rows or merge
// directly into global per-query topK.

extern "C" __global__ void ung_singleton_top1_global_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ q_norm,
    const float* __restrict__ all_x,
    const float* __restrict__ all_norm,
    const uint32_t* __restrict__ singleton_qi,
    const uint32_t* __restrict__ singleton_xoff,
    int singleton_nq,
    int dim,
    int topk,
    int* __restrict__ out_idx,
    float* __restrict__ out_dist)
{
    int sid = blockIdx.x;
    if (sid >= singleton_nq) return;
    uint32_t qi = singleton_qi[sid];
    uint32_t xoff = singleton_xoff[sid];

    const float* q = Q + (size_t)qi * (size_t)dim;
    const float* x = all_x + (size_t)xoff * (size_t)dim;
    float acc = 0.f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        acc += q[d] * x[d];
    }

    __shared__ float sm[256];
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float dist = q_norm[qi] + all_norm[xoff] - 2.f * sm[0];
        size_t base = (size_t)qi * (size_t)topk;
        out_idx[base] = 0;
        out_dist[base] = dist;
    }
}

extern "C" __global__ void ung_singleton_top1_global_merge_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ q_norm,
    const float* __restrict__ all_x,
    const float* __restrict__ all_norm,
    const uint32_t* __restrict__ singleton_qi,
    const uint32_t* __restrict__ singleton_xoff,
    const uint32_t* __restrict__ q_ids,
    int singleton_nq,
    int dim,
    int topk,
    int* __restrict__ global_idx,
    float* __restrict__ global_dist,
    int* __restrict__ locks)
{
    const int sid = blockIdx.x;
    if (sid >= singleton_nq) return;
    const uint32_t qi = singleton_qi[sid];
    const uint32_t xoff = singleton_xoff[sid];

    const float* q = Q + (size_t)qi * (size_t)dim;
    const float* x = all_x + (size_t)xoff * (size_t)dim;
    float acc = 0.f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        acc += q[d] * x[d];
    }

    __shared__ float sm[256];
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const uint32_t qid = q_ids[qi];
        const float dist = q_norm[qi] + all_norm[xoff] - 2.f * sm[0];
        int* lock = locks + qid;
        while (atomicCAS(lock, 0, 1) != 0) { }
        int* out_i = global_idx + (size_t)qid * (size_t)topk;
        float* out_d = global_dist + (size_t)qid * (size_t)topk;
        ung_global_topk_insert_locked(out_i, out_d, topk, (int)xoff, dist);
        __threadfence();
        atomicExch(lock, 0);
    }
}
