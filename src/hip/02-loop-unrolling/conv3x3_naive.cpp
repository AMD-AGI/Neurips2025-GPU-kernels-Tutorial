#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t err = call;                                                 \
        if (err != hipSuccess) {                                               \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__       \
                      << " code=" << static_cast<int>(err)                     \
                      << " \"" << hipGetErrorString(err) << "\"" << std::endl;\
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#ifndef INNER_ITERS
#define INNER_ITERS 64
#endif

// -----------------------------------------------------------------------------
// Naive 3x3 convolution with explicit "no unroll" directives.
// Each thread computes one pixel, repeating the 3×3 convolution INNER_ITERS times.
// -----------------------------------------------------------------------------
__global__ void conv3x3_naive_no_unroll(const float* __restrict__ input,
                                        const float* __restrict__ kernel,
                                        float* __restrict__ output,
                                        int width,
                                        int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Skip the borders.
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;

    int idx = y * width + x;
    float out_acc = 0.0f;

    // Repeat convolution multiple times to make loop overhead significant.
    for (int it = 0; it < INNER_ITERS; ++it) {
        float acc = 0.0f;

        // Explicitly disable unrolling.
        #pragma unroll 1
        for (int ky = -1; ky <= 1; ++ky) {
            #pragma unroll 1
            for (int kx = -1; kx <= 1; ++kx) {
                float v = input[(y + ky) * width + (x + kx)];
                float w = kernel[(ky + 1) * 3 + (kx + 1)];
                acc = fmaf(v, w, acc);
            }
        }

        out_acc += acc;
    }

    // Normalize so magnitude is comparable between kernels.
    output[idx] = out_acc / INNER_ITERS;
}

// -----------------------------------------------------------------------------
// Host-side driver
// -----------------------------------------------------------------------------
int main() {
    const int width  = 4096;
    const int height = 4096;
    const int num_pixels = width * height;
    const int num_runs = 50;

    std::cout << "Running conv3x3_naive_no_unroll on " 
              << width << "x" << height << " image, INNER_ITERS=" 
              << INNER_ITERS << "\n";

    // Host memory.
    std::vector<float> h_input(num_pixels, 1.0f);
    std::vector<float> h_output(num_pixels, 0.0f);

    // Gaussian blur kernel (sum=16, avoids zero output).
    float h_kernel[9] = {
        1, 2, 1,
        2, 4, 2,
        1, 2, 1
    };

    // Device buffers.
    float *d_input = nullptr, *d_output = nullptr, *d_kernel = nullptr;
    HIP_CHECK(hipMalloc(&d_input,  num_pixels * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, num_pixels * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_kernel, 9 * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_input,  h_input.data(),
                        num_pixels * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_kernel, h_kernel,
                        9 * sizeof(float), hipMemcpyHostToDevice));

    HIP_CHECK(hipMemset(d_output, 0, num_pixels * sizeof(float)));

    dim3 block(16, 16);
    dim3 grid((width  + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    // Warm-up run.
    conv3x3_naive_no_unroll<<<grid, block>>>(d_input, d_kernel, d_output, width, height);
    HIP_CHECK(hipDeviceSynchronize());

    // Timing with hipEvent.
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        conv3x3_naive_no_unroll<<<grid, block>>>(d_input, d_kernel, d_output, width, height);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float total_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / num_runs;

    std::cout << "Average time: " << avg_ms << " ms\n";

    // Fetch one sample output to verify correctness.
    HIP_CHECK(hipMemcpy(h_output.data(), d_output,
                        num_pixels * sizeof(float), hipMemcpyDeviceToHost));

    std::cout << "Sample output = "
              << h_output[width * (height/2) + width/2] << "\n";

    // Cleanup.
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_kernel));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    return 0;
}
