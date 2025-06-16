#include <iostream> 
#include <vector> 
#include <chrono> 
#include <omp.h> 

using namespace std; 

void convolution2D( 
    const float* input, 
    const float* kernel, 
    float* output, 
    int input_rows, 
    int input_cols, 
    int kernel_rows, 
    int kernel_cols
) { 
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1; 

    // 8.5 gflops for now
    #pragma omp parallel for collapse(4)
    for (int i = 0; i < output_rows; i++) { 
        for (int j = 0; j < output_cols; j++) { 
            float sum = 0.0f; 
            for (int k = 0; k < kernel_rows; k++) { 
                for (int l = 0; l < kernel_cols; l++) { 
                    sum += input[(i + k) * input_cols + (j + l)] * kernel[(k * kernel_cols) + l];
                }
            }
            output[i * output_cols + j] = sum; 
        }
    }
}

int main() { 
    const int input_rows = 1000; 
    const int input_cols = 1000; 
    const int kernel_rows = 4; 
    const int kernel_cols = 4; 

    float input[input_rows * input_cols]; 
    float kernel[kernel_rows * kernel_cols] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };

    for (int i = 0; i < input_rows * input_cols; i++) { 
        input[i] = rand() % 100; 
    }

    const int output_rows = input_rows - kernel_rows + 1;
    const int output_cols = input_cols - kernel_cols + 1;
    float output[output_rows * output_cols]; 

    long long flops = (long long)input_rows * input_cols * kernel_rows * kernel_cols * 2; 
    
    auto start = std::chrono::high_resolution_clock::now();
    
    convolution2D(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double seconds = elapsed.count();
    
    double gflops = (flops / seconds) / 1e9;
    
    std::cout << "Execution time: " << seconds * 1000 << " ms" << std::endl;
    std::cout << "GFLOPS: " << gflops << std::endl;

}