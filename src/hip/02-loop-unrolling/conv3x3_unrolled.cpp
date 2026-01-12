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
// Manually unrolled 3×3 convolution. No loops in the kernel.
// Repeated INNER_ITERS times to emphasize ILP and remove loop overhead.
// -----------------------------------------------------------------------------
__global__ void conv3x3_unrolled(const float* __restrict__ input,
                                 const float* __restrict__ kernel,
                                 float* __restrict__ output,
                                 int width,
                                 int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;

    int center_idx = y * width + x;
    int row_above  = center_idx - width;
    int row_below  = center_idx + width;

    float out_acc = 0.0f;

    for (int it = 0; it < INNER_ITERS; ++it) {
        float acc = 0.0f;

        // Fully unrolled 3×3 convolution.
        acc = fmaf(input[row_above - 1], kernel[0], acc);
        acc = fmaf(input[row_above + 0], kernel[1], acc);
        acc = fmaf(input[row_above + 1], kernel[2], acc);

        acc = fmaf(input[center_idx - 1], kernel[3], acc);
        acc = fmaf(input[center_idx + 0], kernel[4], acc);
        acc = fmaf(input[center_idx + 1], kernel[5], acc);

        acc = fmaf(input[row_below - 1], kernel[6], acc);
        acc = fmaf(input[row_below + 0], kernel[7], acc);
        acc = fmaf(input[row_below + 1], kernel[8], acc);

        out_acc += acc;
    }

    output[center_idx] = out_acc / INNER_ITERS;
}

// -----------------------------------------------------------------------------
// Host-side driver
// -----------------------------------------------------------------------------
int main() {
    const int width  = 4096;
    const int height = 4096;
    const int num_pixels = width * height;
    const int num_runs = 50;

    std::cout << "Running conv3x3_unrolled on "
              << width << "x" << height << " image, INNER_ITERS="
              << INNER_ITERS << "\n";

    // Host memory.
    std::vector<float> h_input(num_pixels, 1.0f);
    std::vector<float> h_output(num_pixels, 0.0f);

    // Gaussian blur kernel (sum=16).
    float h_kernel[9] = {
        1, 2, 1,
        2, 4, 2,
        1, 2, 1
    };

    // Device memory.
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

    // Warm-up.
    conv3x3_unrolled<<<grid, block>>>(d_input, d_kernel, d_output, width, height);
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        conv3x3_unrolled<<<grid, block>>>(d_input, d_kernel, d_output, width, height);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float total_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / num_runs;

    std::cout << "Average time: " << avg_ms << " ms\n";

    HIP_CHECK(hipMemcpy(h_output.data(), d_output,
                        num_pixels * sizeof(float),
                        hipMemcpyDeviceToHost));

    std::cout << "Sample output = "
              << h_output[width * (height/2) + width/2] << "\n";

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_kernel));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    return 0;
}
