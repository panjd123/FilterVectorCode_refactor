#include <iostream>
#include <cstdlib>
#include <string>
#include "distance.h"


namespace ANNS {

    std::unique_ptr<DistanceHandler> get_distance_handler(const std::string& data_type, const std::string& dist_fn) {
        if (data_type == "float") {
            if (dist_fn == "L2")
                return std::make_unique<FloatL2DistanceHandler>();
            else if (dist_fn == "IP") {
                std::cerr << "Not implement distance function: " << dist_fn << " for data type: " << data_type << std::endl;
                exit(-1);
            } else if (dist_fn == "cosine") {
                std::cerr << "Not implement distance function: " << dist_fn << " for data type: " << data_type << std::endl;
                exit(-1);
            } else {
                std::cerr << "Error: invalid distance function: " << dist_fn << " and data type: " << data_type << std::endl;
                exit(-1);
            }
        } else if (data_type == "int8") {
            std::cerr << "Not implement distance function: " << dist_fn << " for data type: " << data_type << std::endl;
            exit(-1);
        } else if (data_type == "uint8") {
            std::cerr << "Not implement distance function: " << dist_fn << " for data type: " << data_type << std::endl;
            exit(-1);
        } else {
            std::cerr << "Not implement distance function: " << dist_fn << " for data type: " << data_type << std::endl;
            exit(-1);
        }
    }

    // float L2 distance
    FloatL2DistanceHandler::FloatL2DistanceHandler() {
        if (const char *value = std::getenv("UNG_L2_KERNEL")) {
            const std::string kernel(value);
            if (kernel == "avx2" || kernel == "baseline" || kernel == "old")
                kernel_ = Kernel::Avx2;
            else if (kernel == "avx2_fma4" || kernel == "fma4" || kernel == "fma")
                kernel_ = Kernel::Avx2Fma4;
            else
                std::cerr << "[distance] unknown UNG_L2_KERNEL=" << kernel
                          << "; using avx2" << std::endl;
        }
    }

    float FloatL2DistanceHandler::compute(const char *a, const char *b, IdxType dim) const {
        const float *x = reinterpret_cast<const float *>(a);
        const float *y = reinterpret_cast<const float *>(b);
        switch (kernel_) {
            case Kernel::Avx2:
                return compute_avx2(x, y, dim);
            case Kernel::Avx2Fma4:
            default:
                return compute_avx2_fma4(x, y, dim);
        }
    }

    float FloatL2DistanceHandler::compute_avx2(const float *x, const float *y, IdxType dim) {
        __m256 msum0 = _mm256_setzero_ps();

        while (dim >= 8) {
            __m256 mx = _mm256_loadu_ps(x);
            x += 8;
            __m256 my = _mm256_loadu_ps(y);
            y += 8;
            const __m256 a_m_b1 = _mm256_sub_ps(mx, my);
            msum0 = _mm256_add_ps(msum0, _mm256_mul_ps(a_m_b1, a_m_b1));
            dim -= 8;
        }

        __m128 msum1 = _mm256_extractf128_ps(msum0, 1);
        __m128 msum2 = _mm256_extractf128_ps(msum0, 0);
        msum1 = _mm_add_ps(msum1, msum2);

        if (dim >= 4) {
            __m128 mx = _mm_loadu_ps(x);
            x += 4;
            __m128 my = _mm_loadu_ps(y);
            y += 4;
            const __m128 a_m_b1 = _mm_sub_ps(mx, my);
            msum1 = _mm_add_ps(msum1, _mm_mul_ps(a_m_b1, a_m_b1));
            dim -= 4;
        }

        if (dim > 0) {
            __m128 mx = masked_read(dim, x);
            __m128 my = masked_read(dim, y);
            __m128 a_m_b1 = _mm_sub_ps(mx, my);
            msum1 = _mm_add_ps(msum1, _mm_mul_ps(a_m_b1, a_m_b1));
        }

        msum1 = _mm_hadd_ps(msum1, msum1);
        msum1 = _mm_hadd_ps(msum1, msum1);
        return _mm_cvtss_f32(msum1);
    }

    float FloatL2DistanceHandler::compute_avx2_fma4(const float *x, const float *y, IdxType dim) {
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        while (dim >= 32) {
            __m256 dx0 = _mm256_sub_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y));
            __m256 dx1 = _mm256_sub_ps(_mm256_loadu_ps(x + 8), _mm256_loadu_ps(y + 8));
            __m256 dx2 = _mm256_sub_ps(_mm256_loadu_ps(x + 16), _mm256_loadu_ps(y + 16));
            __m256 dx3 = _mm256_sub_ps(_mm256_loadu_ps(x + 24), _mm256_loadu_ps(y + 24));
            sum0 = _mm256_fmadd_ps(dx0, dx0, sum0);
            sum1 = _mm256_fmadd_ps(dx1, dx1, sum1);
            sum2 = _mm256_fmadd_ps(dx2, dx2, sum2);
            sum3 = _mm256_fmadd_ps(dx3, dx3, sum3);
            x += 32;
            y += 32;
            dim -= 32;
        }

        sum0 = _mm256_add_ps(sum0, sum1);
        sum2 = _mm256_add_ps(sum2, sum3);
        sum0 = _mm256_add_ps(sum0, sum2);

        while (dim >= 8) {
            __m256 dx = _mm256_sub_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y));
            sum0 = _mm256_fmadd_ps(dx, dx, sum0);
            x += 8;
            y += 8;
            dim -= 8;
        }

        __m128 lo = _mm256_castps256_ps128(sum0);
        __m128 hi = _mm256_extractf128_ps(sum0, 1);
        __m128 sum128 = _mm_add_ps(lo, hi);

        if (dim >= 4) {
            __m128 dx = _mm_sub_ps(_mm_loadu_ps(x), _mm_loadu_ps(y));
            sum128 = _mm_add_ps(sum128, _mm_mul_ps(dx, dx));
            x += 4;
            y += 4;
            dim -= 4;
        }

        if (dim > 0) {
            __m128 dx = _mm_sub_ps(masked_read(dim, x), masked_read(dim, y));
            sum128 = _mm_add_ps(sum128, _mm_mul_ps(dx, dx));
        }

        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        return _mm_cvtss_f32(sum128);
    }

    __m128 FloatL2DistanceHandler::masked_read(IdxType dim, const float *x) {
        __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
        switch (dim) {
            case 3:
                buf[2] = x[2];
            case 2:
                buf[1] = x[1];
            case 1:
                buf[0] = x[0];
        }
        return _mm_load_ps(buf);
    }
}
