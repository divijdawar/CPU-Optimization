#include <cstdlib>
#include <ctime>
#include <cstdio>

/*
So far, we had achieved ~86.30 gflop/s using multithreaded tiling.
Here we acheive ~231.60 gflop/s
*/
template <int T>
void naiveIterativeMatmulTiled(
        float* const A,
        float* const B,
        float* const C,
        const int M,
        const int N,
        const int K) {

    for (int i = 0; i < M * N; i++){
        C[i] = 0.0f;
    }

    for (int m = 0; m < M/T; m += T) {
        for (int n = 0; n < N/T; n += T) {
            for (int k = 0; k < K/T; k += T) {
                for (int mt = m; mt < m + T && mt < M; mt++) {
                    for (int nt = n; nt < n + T && nt < N; nt++) {
                        for (int kt = k; kt < k + T && kt < K; kt++) {
                            C[mt * M + nt] += A[mt * M + kt] * B[kt * K + nt];
                        }
                    }
                }
            }
        }
    }
}

int main() {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    float *C = (float *)malloc(M * N * sizeof(float));
    
    for (int i = 0; i < M * K; i++) {
        A[i] = 1.0f;
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = 2.0f;
    }
    
    int speed = 0;
    
    for(int i = 0; i < 5; i++){
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        naiveIterativeMatmulTiled<128>(A, B, C, M, N, K);
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
        double gflops = (2.0 * M * N * K) * 1e-9 / elapsed;
        speed += gflops;
    }

    printf("Average GFLOPS: %.2f\n", (double)speed / 5.0); // ~ 231.60 gflop/s
    
    free(A);
    free(B);
    free(C);
    return 0;
}