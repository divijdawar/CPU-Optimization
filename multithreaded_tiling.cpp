#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <omp.h>

/*
We achieved ~ 6.09 gflop/s previously,
with this code we've achieved ~ 86.30 gflop/s a 14x improvement
*/  

template <int rows, int columns, int inners, int tileSize> 
inline void tiled_matmul(const float *left, const float *right, float *result) {
    int n = rows;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i * n + j] = 0.0;
        }
    }
    
    #pragma omp parallel for shared(result, left, right) default(none) collapse(2) num_threads(8)
    for (int rowTile = 0; rowTile < rows;  rowTile += tileSize) {
        for (int columnTile = 0; columnTile < columns; columnTile += tileSize) {
            for (int innerTile = 0; innerTile < inners; innerTile += tileSize) {
                for (int row = rowTile; row < rowTile + tileSize; row++) {
                    int innerTileEnd = std::min(inners, innerTile + tileSize);
                    for (int inner = innerTile; inner < innerTileEnd; inner++) {
                        #pragma omp simd
                        for (int col = columnTile; col < columnTile + tileSize; col++) {
                            result[row * columns + col] += left[row * inners + inner] * right[inner * columns + col];
                        }
                    }
                }
            }
        }
    }
}
  

int main() {

    const int N = 1024; 
    const int M = 1024; 
    const int K = 1024; 

    float *left = (float *)malloc(N * K * sizeof(float));
    float *right = (float *)malloc(K * M * sizeof(float));
    float *result = (float *)malloc(N * M * sizeof(float));

    for (int i = 0; i < N * K; i++) {
        left[i] = 1.0f;
    }

    for (int i = 0; i < K * M; i++) {
        right[i] = 2.0f;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    tiled_matmul<1024, 1024, 1024, 128>(left, right, result);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
    double gflops = (2.0 * N * M * K) * 1e-9 / elapsed;
    printf("GFLOPS: %.2f\n", gflops); 
    printf("Execution time: %.2f seconds\n", elapsed);

    free(left);
    free(right);
    free(result);
    return 0;
}