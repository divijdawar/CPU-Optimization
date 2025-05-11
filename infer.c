#include <arm_neon.h>
#include <assert.h>
#include <stdint.h>
#include <omp.h>

typedef uint16_t, f16_t;

//matmul supporting float16 weights on ARM 
static void matmul(
    float *out, 
    const float *x, 
    const float *f16_t *w, 
    int n, 
    int d
) {
    #if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    // accumulating 8x8 matrix multiplication
    assert(n % 8 == 0);

    #pragma omp parallel for
    for (int i = 0; i < d; i += 8) {
        float32x4_t sum_lo = vdupq_n_f32(0.0f);
        float32x4_t sum_hi = vdupq_n_f32(0.0f);

        // Extracts the ith row of the weight matrix
        const float16x8_t *wrow = (const float16_t*)&w[i * n];

        for (int j = 0; j < n; j += 8) {
            // Load 8 fp16 values and convert to fp32
            float16x8_t   w16   = vld1q_f16(&wrow[j]);  // loads 8 f16 values from wrow[j] into w16        
            float16x4_t   w16lo = vget_low_f16 (w16);   // returns the lower 4 f16 values from w16
            float16x4_t   w16hi = vget_high_f16(w16);   // returns the upper 4 f16 values from w16
            float32x4_t   w32lo = vcvt_f32_f16(w16lo);  // converts the lower 4 f16 values from w16 to f32
            float32x4_t   w32hi = vcvt_f32_f16(w16hi);  // converts the upper 4 f16 values from w16 to f32

            float32x4_t x32lo = vld1q_f32(&x[j]);
            float32x4_t x32hi = vld1q_f32(&x[j + 4]);

            // Fused multiply-add
            sum_lo = vfmaq_f32(sum_lo, x32lo, w32lo);
            sum_hi = vfmaq_f32(sum_hi, x32hi, w32hi);

            float32x4_t sum = vaddq_f32(sum_lo, sum_hi);
        }
        
        #if defined(__aarch64__) && __ARM_ARCH >=8
            float result = vaddvq_f32(sum);
        #else 
            float32x2_2 tmp = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
            float result = vget_lane_f32(vpadd_f32(tmp, tmp), 0);
        #endif
            out[i] = result;
    }
    #endif
}
