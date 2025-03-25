#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

// We've achieved ~0.65 glop/s so far, lets try and improve that 
// Using tiling, we've achieved ~ 6.09 gflop/s when tileSize = 128, rougly a 9x improvement

template <int rows, int columns, int inners, int tileSize> 

inline void tiled_matmul(const float *left, const float *right, float *result) {

    // Initialize result matrix to zeros
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            result[i*columns+j] = 0.0f;
        }
    }
    
    for (int i = 0; i < inners; i+=tileSize) {
        for (int j =0; j < rows; j++) {
            int innerTileEnd = std::min(inners, i+tileSize);
            for (int inner = 0; inner < innerTileEnd; inner++) {
                for (int column = 0; column < columns; column++) {
                    result[j*columns+column] += left[j*inners+inner]*right[inner*columns+column];
                }
            }
        }
    }
}

int main(){
    const int N =  1024;
    const int M =  1024;
    const int K =  1024;
    float *left = (float *)malloc(N * K * sizeof(float));
    float *right = (float *)malloc(K * M * sizeof(float));
    float *result = (float *)malloc(N * M * sizeof(float));

    for (int i = 0; i < N * K; i++){
        left[i] = 1.0f;
        right[i] = 2.0f;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    tiled_matmul<1024, 1024, 1024, 128>(left, right, result);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
    double gflops = (2.0 * N * M * K) * 1e-9 / elapsed;
    printf("GFLOPS: %.2f\n", gflops); // ~ 3.36 gflop/s when tileSize = 64, ~ 6.09 gflop/s when tileSize = 128
    printf("Execution time: %.2f seconds\n", elapsed);

    free(left);
    free(right);
    free(result);
    return 0;
}