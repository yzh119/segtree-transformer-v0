/* TODOs
 * - segment_reduce_forward, segment_reduce_backward;
 * - switch backend from aten to dlpack
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Type.h>
#include <c10/util/Exception.h>
#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define AT_CASE_ITYPE(enum_type, type, DTYPE, NAME, ...)                    \
  case enum_type: {                                                         \
    const at::Type& dtype = DTYPE;                                          \
    using idx_t = type;                                                     \
    switch (dtype.scalarType()) {                                           \
      case at::ScalarType::Half: {                                          \
        using data_t = at::Half;                                            \
        return __VA_ARGS__();                                               \
      }                                                                     \
      case at::ScalarType::Float: {                                         \
        using data_t = float;                                               \
        return __VA_ARGS__();                                               \
      }                                                                     \
      case at::ScalarType::Double: {                                        \
        using data_t = double;                                              \
        return __VA_ARGS__();                                               \
      }                                                                     \
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", dtype.toString(), "'");   \
    }                                                                       \
  }                

#define AT_DISPATCH_IDX_DATA_TYPES(ITYPE, DTYPE, NAME, ...)                             \
  [&] {                                                                                 \
    const at::Type& itype = ITYPE;                                                      \
    switch (itype.scalarType()) {                                                       \
      AT_CASE_ITYPE(at::ScalarType::Int, int32_t, DTYPE, NAME, __VA_ARGS__)             \
      AT_CASE_ITYPE(at::ScalarType::Long, int64_t, DTYPE, NAME, __VA_ARGS__)            \
      default:                                                                          \
        AT_ERROR(#NAME, " not implemented for '", itype.toString(), "'");               \
    }                                                                                   \
  }()

namespace {

/*
 * CUDA Kernel of the forward function for Masked Matrix Multiplication:
 * y = adj * (A @ B^T)
 * This is an unoptimized version, to better utilize shared memory, some sort of padding is required.
 * Note that we use the row and col vector to represent the sparse matrix adj. (coo format)
 */
template <typename idx_t, typename data_t>
__global__ void maskedmm_forward_kernel(const idx_t* __restrict__ row, const idx_t* __restrict__ col, const data_t* __restrict__ A, const data_t* __restrict__ Bt, data_t* __restrict__ y, const int e, const int d, const int n, const int h) {
    int i = (((blockIdx.x) * blockDim.x) + (threadIdx.x));
    if (i < e) {
        for (int ko = 0; ko < h; ++ko) {
            data_t sum = 0;
            for (int k = 0; k < d; ++k) {
                sum += A[(row[i] * h + ko) * d + k] * Bt[col[i] + ((ko * d + k) * n)];
            }
            y[i * h + ko] = sum;
        }
    }
}

/*
 * CUDA Kernel of the backward function for Masked Matrix Multiplication: 
 * dA = B @ (dy * adj)
 * dB = A @ (dy * adj)
 * Mostly the same as src_mul_edge
 */
template <typename idx_t, typename data_t>
__global__ void maskedmm_backward_kernel(const idx_t* __restrict__ row, const idx_t* __restrict__ col, const data_t* __restrict__ A, const data_t* __restrict__ B, const data_t* __restrict__ dy, data_t* __restrict__ dA, data_t* __restrict__ dB, const int e, const int d, const int n, const int h) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < d * h) {
        for (int k = 0; k < n; ++k) {
            dA[k * d * h + j] = 0;
            dB[k * d * h + j] = 0;
        }
        for (int k = 0; k < e; ++k) {
            dA[row[k] * d * h + j] += dy[k * h + j / d] * B[col[k] * d * h + j];
            dB[col[k] * d * h + j] += dy[k * h + j / d] * A[row[k] * d * h + j];
        }
    }
}

/*
 * CUDA Kernel of the forward function for Node-Edge Multiplication(reduced on edge, designed for relative positional encoding).
 */
template <typename idx_t, typename data_t>
__global__ void node_mul_edge_forward_kernel(const idx_t* __restrict__ indptr, const idx_t* __restrict__ eid, const data_t* __restrict__ A, const data_t* __restrict__ B, data_t* __restrict__ y, const int d, const int n, const int h) {
    int i = blockIdx.x;
    int tx = threadIdx.x;
    if (i < n) {
        for (int j = indptr[i] + tx; j < indptr[i + 1]; j += blockDim.x)
            for (int ko = 0; ko < h; ++ko) {
                data_t sum = 0;
                for (int ki = 0; ki < d; ++ki) {
                    sum += A[(i * h + ko) * d + ki] * B[eid[j] * d + ki];
                }
                y[eid[j] * h + ko] = sum;
            }
    }
}


/*
 * CUDA Kernel of the forward function for Masked Matrix Multiplication. (argument: csr format)
 */
template <typename idx_t, typename data_t>
__global__ void maskedmm_csr_forward_kernel(const idx_t* __restrict__ indptr, const idx_t* __restrict__ eid, const idx_t* __restrict__ indices, const data_t* __restrict__ A, const data_t* __restrict__ B, data_t* __restrict__ y, const int d, const int n, const int h) {
    int i = blockIdx.x; 
    int tx = threadIdx.x;
    if (i < n) {
        for (int j = indptr[i] + tx; j < indptr[i + 1]; j += blockDim.x)
            for (int ko = 0; ko < h; ++ko) {
                data_t sum = 0;
                for (int ki = 0; ki < d; ++ki) {
                    sum += A[(i * h + ko) * d + ki] * B[(ko * d + ki) * n + indices[j]];
                }
                y[eid[j] * h + ko] = sum;
            }
    }
}


/*
 * CUDA Kernel of the backward function for Node-Edge Multiplication(reduced on edge, designed for relative positional encoding).
 */
template <typename idx_t, typename data_t>
__global__ void node_mul_edge_backward_kernel_0(const idx_t* __restrict__ indptr, const idx_t* __restrict__ eid, const data_t* __restrict__ B, const data_t* __restrict__ dy, data_t* __restrict__ dA, const int d, const int n, const int h) {
    int tx = threadIdx.x;
    int i = blockIdx.x;
    if (i < n) {
        for (int j = tx; j < d * h; j += blockDim.x) {
            data_t sum = 0;
            for (int k = indptr[i]; k < indptr[i + 1]; ++k)
                sum += dy[eid[k] * h + j / d] * B[eid[k] * d + j % d];
            dA[i * d * h + j] = sum; 
        }
    }
}


/*
 * CUDA Kernel of the backward function for Node-Edge Multiplication(reduced on edge, designed for relative positional encoding).
 */
template <typename idx_t, typename data_t>
__global__ void node_mul_edge_backward_kernel_1(const idx_t* __restrict__ indptr, const idx_t* __restrict__ eid, const data_t* __restrict__ A, const data_t* __restrict__ dy, data_t* __restrict__ dB, const int d, const int n, const int h) {
    int tx = threadIdx.x;
    int i = blockIdx.x;
    if (i < n) {
        for (int j = tx; j < d; j += blockDim.x) {
            for (int k = indptr[i]; k < indptr[i + 1]; ++k) {
                data_t sum = 0;
                for (int ki = 0; ki < h; ++ki) {
                    sum += dy[eid[k] * h + ki] * A[(i * h + ki) * d + j];
                }
                dB[eid[k] * d + j] = sum;
            }
        }
    }
}


/*
 * CUDA Kernel of the backward function for Masked Matrix Multiplication. (argument: csr format)
 */
template <typename idx_t, typename data_t>
__global__ void maskedmm_csr_backward_kernel(const idx_t* __restrict__ indptr_r, const idx_t* __restrict__ eid_r, const idx_t* __restrict__ indices_r, const idx_t* __restrict__ indptr_c, const idx_t* __restrict__ eid_c, const idx_t* __restrict__ indices_c, const data_t* __restrict__ A, const data_t* __restrict__ B, const data_t* __restrict__ dy, data_t* __restrict__ dA, data_t* __restrict__ dB, const int d, const int n, const int h) {
    int tx = threadIdx.x;
    int i = blockIdx.x;
    if (i < n) {
        for (int j = tx; j < d * h; j += blockDim.x) {
            data_t sum = 0;
            for (int k = indptr_r[i]; k < indptr_r[i + 1]; ++k)
                sum += dy[eid_r[k] * h + j / d] * B[indices_r[k] * d * h + j];
            dA[i * d * h + j] = sum;

            sum = 0;
            for (int k = indptr_c[i]; k < indptr_c[i + 1]; ++k)
                sum += dy[eid_c[k] * h + j / d] * A[indices_c[k] * d * h + j];
            dB[i * d * h + j] = sum;
        }
    }
}

/*
 * CUDA Kernel of the forward function for Source Multiply Edge Function.
 * For `src_mul_edge` operation, the arguments are csr(column-major) representations.
 */
template <typename idx_t, typename data_t>
__global__ void vector_spmm_forward_kernel(const idx_t* __restrict__ indptr, const idx_t* __restrict__ eid, const idx_t* __restrict__ indices, const data_t* __restrict__ edata, const data_t* __restrict__ x, data_t* __restrict__ y, const int d, const int n, const int h) {
    int i = blockIdx.x;
    int tx = threadIdx.x;
    if (i < n) {
        for (int j = tx; j < d * h; j += blockDim.x) {
            data_t sum = 0;
            for (int k = indptr[i]; k < indptr[i + 1]; ++k)
                sum += edata[eid[k] * h + j / d] * x[indices[k] * d * h + j];
            y[i * d * h + j] = sum;
        }
    }
}

/*
 * CUDA Kernel of the backward function for Source Multiply Edge Function.
 */
template <typename idx_t, typename data_t>
__global__ void vector_spmm_backward_kernel_0(const idx_t* __restrict__ indptr, const idx_t* __restrict__ eid, const idx_t* __restrict__ indices, const data_t* __restrict__ dy, const data_t* __restrict__ xt, data_t* __restrict__ dedata, const int d, const int n, const int h) {
    int i = blockIdx.x; 
    int tx = threadIdx.x;
    if (i < n) {
        for (int j = indptr[i] + tx; j < indptr[i + 1]; j += blockDim.x)
            for (int ko = 0; ko < h; ++ko) {
                data_t sum = 0;
                for (int ki = 0; ki < d; ++ki) {
                    sum += dy[(i * h + ko) * d + ki] * xt[(ko * d + ki) * n + indices[j]];
                }
                dedata[eid[j] * h + ko] = sum;
            }
    }
}

template <typename idx_t, typename data_t>
__global__ void vector_spmm_backward_kernel_1(const idx_t* __restrict__ indptr, const idx_t* __restrict__ eid, const idx_t* __restrict__ indices, const data_t* __restrict__ edata, const data_t* __restrict__ dy, data_t* __restrict__ dx, const int d, const int n, const int h) {
    int i = blockIdx.x; 
    int tx = threadIdx.x;
    if (i < n) {
        for (int j = tx; j < d * h; j += blockDim.x) {
            data_t sum = 0;
            for (int k = indptr[i]; k < indptr[i + 1]; ++k)
                sum += edata[eid[k] * h + j / d] * dy[indices[k] * d * h + j];
            dx[i * d * h + j] = sum;
        }
    }
}

/*
 * CUDA Kernel of forward function for Sparse Softmax
 * y = softmax(x), grouped by node.
 * indptr, eid: csr format
 */
template <typename idx_t, typename data_t>
__global__ void sparse_softmax_forward_kernel(const idx_t* __restrict__ indptr, const idx_t* __restrict__ eid, const data_t* __restrict__ x, data_t* __restrict__ y, const int n, const int h) {
    //data_t max_val = -1e9;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = threadIdx.y;
    if (i < n) {
        data_t max_val = (indptr[i] < indptr[i + 1]) ? x[eid[indptr[i]] * h + j]: (data_t)(0);
        for (int k = indptr[i]; k < indptr[i + 1]; ++k)
            max_val = max(max_val, x[eid[k] * h + j]);

        data_t sum = 0;
        for (int k = indptr[i]; k < indptr[i + 1]; ++k) {
            data_t now = exp(x[eid[k] * h + j] - max_val);
            y[eid[k] * h + j] = now;
            sum += now;
        }

        for (int k = indptr[i]; k < indptr[i + 1]; ++k)
            y[eid[k] * h + j] /= sum;
    }
}

/*
 * CUDA Kernel of backward function for Sparse Softmax.
 * indptr, eid: csr format
 */
template <typename idx_t, typename data_t>
__global__ void sparse_softmax_backward_kernel(const idx_t* __restrict__ indptr, const idx_t* __restrict__ eid, const data_t* __restrict__ dy, const data_t* __restrict__ y, data_t* __restrict__ dx, const int n, const int h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    if (i < n) {
        for (int kj = indptr[i] + ty; kj < indptr[i + 1]; kj += blockDim.y) {
            data_t dsum = 0;
            for (int ki = indptr[i]; ki < indptr[i + 1]; ++ki) {
                dsum -= dy[eid[ki] * h + tz] * y[eid[ki] * h + tz] * y[eid[kj] * h + tz];
                if (ki == kj) dsum += dy[eid[ki] * h + tz] * y[eid[ki] * h + tz];
            }
            dx[eid[kj] * h + tz] = dsum;
        }
    }
}

} // End of namespace

at::Tensor maskedmm_cuda_forward(
    const at::Tensor& row,
    const at::Tensor& col,
    const at::Tensor& A,
    const at::Tensor& B) {
    // row, col: (e); A, B: (n, d) or (n, h, d); y: (e, h)
    cudaSetDevice(row.get_device());

    const auto e = row.size(0);
    const auto n = A.size(0);
    const auto d = A.size(-1);
    const auto h = (A.dim() == 2) ? 1: A.size(1);
    auto y = (h == 1) ? at::zeros({e}, A.options()): at::zeros({e, h}, A.options());

    const int threads = 1024;
    const dim3 blocks((e + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto Bt = (h == 1) ? B.transpose(0, 1).contiguous(): B.permute({1, 2, 0}).contiguous();

    AT_DISPATCH_IDX_DATA_TYPES(row.type(), A.type(), "maskedmm_cuda_forward", ([&] {
        maskedmm_forward_kernel<idx_t, data_t><<<blocks, threads, 0, stream>>>(
            row.data<idx_t>(),
            col.data<idx_t>(),
            A.data<data_t>(),
            Bt.data<data_t>(),
            y.data<data_t>(),
            e, d, n, h);
    }));
    THCudaCheck(cudaGetLastError());
    return y;
}

std::vector<at::Tensor> maskedmm_cuda_backward(
    const at::Tensor& row,
    const at::Tensor& col,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& dy) {
    // row, col: (e); dy: (e) or (e, h); A, B: (n, d) or (n, h, d);
    cudaSetDevice(row.get_device());

    const auto e = row.size(0);
    const auto n = A.size(0);
    const auto d = A.size(-1);
    const auto h = (dy.dim() == 2) ? dy.size(1): 1;

    const int threads = 1024; 
    const dim3 blocks((d + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto dA = at::zeros_like(A, A.options());
    auto dB = at::zeros_like(B, B.options());

    AT_DISPATCH_IDX_DATA_TYPES(row.type(), A.type(), "maskedmm_cuda_backward", ([&] {
        maskedmm_backward_kernel<idx_t, data_t><<<blocks, threads, 0, stream>>>(
            row.data<idx_t>(),
            col.data<idx_t>(),
            A.data<data_t>(),
            B.data<data_t>(),
            dy.data<data_t>(),
            dA.data<data_t>(),
            dB.data<data_t>(),
            e, d, n, h);
    }));
    THCudaCheck(cudaGetLastError());
    return {dA, dB};
}

at::Tensor node_mul_edge_cuda_forward(
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& A,
    const at::Tensor& B) {
    // indptr: (n + 1); eid: (e); A: (n, d) or (n, h, d); B: (e, d);
    cudaSetDevice(indptr.get_device());

    const auto e = eid.size(0);
    const auto n = A.size(0);
    assert(n == indptr.size(0) - 1);
    const auto d = A.size(-1);
    const auto h = (A.dim() == 2) ? 1: A.size(1);
    auto y = (h == 1) ? at::zeros({e}, A.options()): at::zeros({e, h}, A.options());

    const int threads = 32;
    const dim3 blocks(n);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_IDX_DATA_TYPES(indptr.type(), A.type(), "node_mul_edge_cuda_forward", ([&] {
        node_mul_edge_forward_kernel<idx_t, data_t><<<blocks, threads, 0, stream>>>(
            indptr.data<idx_t>(),
            eid.data<idx_t>(),
            A.data<data_t>(),
            B.data<data_t>(),
            y.data<data_t>(),
            d, n, h);
    }));
    THCudaCheck(cudaGetLastError());
    return y;
}

// __global__ void maskedmm_csr_forward_kernel(idx_t* __restrict__ indptr, idx_t* __restrict__ eid, idx_t* __restrict__ indices, data_t* __restrict__ A, data_t* __restrict__ B, data_t* __restrict__ y, int d, int n) {
at::Tensor maskedmm_csr_cuda_forward(
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& indices,
    const at::Tensor& A,
    const at::Tensor& B) {
    // indptr: (n + 1); eid, indices: (e); A, B: (n, d) or (n, h, d); 
    cudaSetDevice(indptr.get_device());

    const auto e = eid.size(0);
    const auto n = A.size(0);
    assert(n == indptr.size(0) - 1);
    const auto d = A.size(-1);
    const auto h = (A.dim() == 2) ? 1: A.size(1);
    auto y = (h == 1) ? at::zeros({e}, A.options()): at::zeros({e, h}, A.options());

    const int threads = 32;
    const dim3 blocks(n);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto Bt = (B.dim() == 2) ? B.transpose(0, 1).contiguous(): B.permute({1, 2, 0}).contiguous();

    AT_DISPATCH_IDX_DATA_TYPES(indptr.type(), A.type(), "maskedmm_csr_cuda_forward", ([&] {
        maskedmm_csr_forward_kernel<idx_t, data_t><<<blocks, threads, 0, stream>>>(
            indptr.data<idx_t>(),
            eid.data<idx_t>(),
            indices.data<idx_t>(),
            A.data<data_t>(),
            Bt.data<data_t>(),
            y.data<data_t>(),
            d, n, h);
    }));
    THCudaCheck(cudaGetLastError());
    return y;
}

std::vector<at::Tensor> node_mul_edge_cuda_backward(
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& dy) {
    // indptr: (n + 1); eid: (e); dy: (e) or (e, h); A: (n, d) or (n, h, d); B: (e, d)
    cudaSetDevice(indptr.get_device());

    const auto e = eid.size(0);
    const auto n = A.size(0);
    const auto d = A.size(-1);
    const auto h = (dy.dim() == 2) ? dy.size(1): 1;

    int threads = 128;
    const dim3 blocks(n);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto dA = at::zeros_like(A, A.options());
    auto dB = at::zeros_like(B, B.options());

    AT_DISPATCH_IDX_DATA_TYPES(indptr.type(), A.type(), "node_mul_edge_cuda_backward_0", ([&] {
        node_mul_edge_backward_kernel_0<idx_t, data_t><<<blocks, threads, 0, stream>>>(
            indptr.data<idx_t>(),
            eid.data<idx_t>(),
            B.data<data_t>(),
            dy.data<data_t>(),
            dA.data<data_t>(),
            d, n, h);
    }));
    threads = d;
    AT_DISPATCH_IDX_DATA_TYPES(indptr.type(), A.type(), "node_mul_edge_cuda_backward_1", ([&] {
        node_mul_edge_backward_kernel_1<idx_t, data_t><<<blocks, threads, 0, stream>>>(
            indptr.data<idx_t>(),
            eid.data<idx_t>(),
            A.data<data_t>(),
            dy.data<data_t>(),
            dB.data<data_t>(),
            d, n, h);
    }));
    THCudaCheck(cudaGetLastError());
    return {dA, dB};
}


// __global__ void maskedmm_csr_backward_kernel(idx_t* __restrict__ indptr_r, idx_t* __restrict__ eid_r, idx_t* __restrict__ indices_r, idx_t* __restrict__ indptr_c, idx_t* __restrict__ eid_c, idx_t* __restrict__ indices_c, data_t* __restrict__ A, data_t* __restrict__ B, data_t* __restrict__ dy, data_t* __restrict__ dA, data_t* __restrict__ dB, int d, int n)
std::vector<at::Tensor> maskedmm_csr_cuda_backward(
    const at::Tensor& indptr_r,
    const at::Tensor& eid_r,
    const at::Tensor& indices_r,
    const at::Tensor& indptr_c,
    const at::Tensor& eid_c,
    const at::Tensor& indices_c,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& dy) {
    // indptr_r, indptr_c: (n + 1); eid_r, eid_c, indices_r, indices_c: (e); dy: (e) or (e, h); A, B: (n, d) or (n, h, d)
    cudaSetDevice(indptr_r.get_device());

    const auto e = eid_r.size(0);
    const auto n = A.size(0);
    const auto d = A.size(-1);
    const auto h = (dy.dim() == 2) ? dy.size(1): 1;

    const int threads = 128;
    const dim3 blocks(n);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto dA = at::zeros_like(A, A.options());
    auto dB = at::zeros_like(B, B.options());

    AT_DISPATCH_IDX_DATA_TYPES(indptr_r.type(), A.type(), "maskedmm_csr_cuda_backward", ([&] {
        maskedmm_csr_backward_kernel<idx_t, data_t><<<blocks, threads, 0, stream>>>(
            indptr_r.data<idx_t>(),
            eid_r.data<idx_t>(),
            indices_r.data<idx_t>(),
            indptr_c.data<idx_t>(),
            eid_c.data<idx_t>(),
            indices_c.data<idx_t>(),
            A.data<data_t>(),
            B.data<data_t>(),
            dy.data<data_t>(),
            dA.data<data_t>(),
            dB.data<data_t>(),
            d, n, h);
    }));
    THCudaCheck(cudaGetLastError());
    return {dA, dB};
}

at::Tensor sparse_softmax_cuda_forward(
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& x) {
    cudaSetDevice(indptr.get_device());

    // indptr: (n + 1); eid: (e); x: (e) or (e, h);
    const auto n = indptr.size(0) - 1;
    const auto h = (x.dim() == 2) ? x.size(1): 1;
    assert(h <= 32);
    const dim3 threads(32, h);
    const dim3 blocks((n + threads.x - 1) / threads.x);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    const auto y = at::zeros_like(x, x.options());
    
    AT_DISPATCH_IDX_DATA_TYPES(eid.type(), x.type(), "sparse_softmax_cuda_forward",([&] {
        sparse_softmax_forward_kernel<idx_t, data_t><<<blocks, threads, 0, stream>>>(
            indptr.data<idx_t>(),
            eid.data<idx_t>(),
            x.data<data_t>(),
            y.data<data_t>(),
            n, h);
    }));
    THCudaCheck(cudaGetLastError());
    return y;
}

at::Tensor sparse_softmax_cuda_backward(
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& y,
    const at::Tensor& dy) {
    cudaSetDevice(indptr.get_device());

    // indptr: (n + 1); eid: (e); y: (e) or (e, h); dy: (e) or (e, h);
    const auto n = indptr.size(0) - 1;
    const auto h = (dy.dim() == 2) ? dy.size(1): 1;
    assert(h <= 32);
    const dim3 threads(1, 32, h);
    const dim3 blocks((n + threads.x - 1) / threads.x);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    const auto dx = at::zeros_like(dy, dy.options());

    AT_DISPATCH_IDX_DATA_TYPES(eid.type(), y.type(), "sparse_softmax_cuda_backward", ([&] {
        sparse_softmax_backward_kernel<idx_t, data_t><<<blocks, threads, 0, stream>>>(
            indptr.data<idx_t>(),
            eid.data<idx_t>(),
            dy.data<data_t>(),
            y.data<data_t>(),
            dx.data<data_t>(),
            n, h); 
    }));
    THCudaCheck(cudaGetLastError());
    return dx;
}

at::Tensor vector_spmm_cuda_forward(
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& indices,
    const at::Tensor& edata,
    const at::Tensor& x) {
    // indptr: (n + 1); eid, indices: (e); edata: (e) or (e, h); x: (n, d) or (n, h, d);
    cudaSetDevice(indptr.get_device());

    const auto n = indptr.size(0) - 1;
    const auto h = (edata.dim() == 2) ? edata.size(1): 1;
    const auto d = x.size(-1); 
    
    const int threads = 128;
    const dim3 blocks(n);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const auto y = at::zeros_like(x, x.options());
    
    AT_DISPATCH_IDX_DATA_TYPES(eid.type(), x.type(), "vector_spmm_forward", ([&] {
        vector_spmm_forward_kernel<idx_t, data_t><<<blocks, threads, 0, stream>>>(
            indptr.data<idx_t>(),
            eid.data<idx_t>(),
            indices.data<idx_t>(),
            edata.data<data_t>(),
            x.data<data_t>(),
            y.data<data_t>(),
            d, n, h);
    }));
    THCudaCheck(cudaGetLastError());
    return y;
}

std::vector<at::Tensor> vector_spmm_cuda_backward(
    const at::Tensor& indptr,
    const at::Tensor& eid,
    const at::Tensor& indices,
    const at::Tensor& indptr_t,
    const at::Tensor& eid_t,
    const at::Tensor& indices_t,
    const at::Tensor& edata,
    const at::Tensor& dy,
    const at::Tensor& x) {
    // indptr: (n + 1); eid, indices: (e); edata: (e) or (e, h); dy, x: (n, d) or (n, h, d); 
    cudaSetDevice(indptr.get_device());

    const auto n = indptr.size(0) - 1;
    const auto h = (edata.dim() == 2) ? edata.size(1): 1;
    const auto d = x.size(-1);

    int threads = 32;
    const dim3 blocks(n);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    const auto xt = (h == 1) ? x.transpose(0, 1).contiguous(): x.permute({1, 2, 0}).contiguous();

    const auto dx = at::zeros_like(x, x.options());
    const auto dedata = at::zeros_like(edata, edata.options());

    AT_DISPATCH_IDX_DATA_TYPES(eid.type(), x.type(), "vector_spmm_backward_0", ([&] {
        vector_spmm_backward_kernel_0<idx_t, data_t><<<blocks, threads, 0, stream>>>(
            indptr.data<idx_t>(),
            eid.data<idx_t>(),
            indices.data<idx_t>(),
            dy.data<data_t>(),
            xt.data<data_t>(),
            dedata.data<data_t>(),
            d, n, h);
    }));

    threads = 128;
    AT_DISPATCH_IDX_DATA_TYPES(eid.type(), x.type(), "vector_spmm_backward_1", ([&] {
        vector_spmm_backward_kernel_1<idx_t, data_t><<<blocks, threads, 0, stream>>>(
            indptr_t.data<idx_t>(),
            eid_t.data<idx_t>(),
            indices_t.data<idx_t>(),
            edata.data<data_t>(),
            dy.data<data_t>(),
            dx.data<data_t>(),
            d, n, h);
    }));
    THCudaCheck(cudaGetLastError());
    return {dedata, dx};
}
