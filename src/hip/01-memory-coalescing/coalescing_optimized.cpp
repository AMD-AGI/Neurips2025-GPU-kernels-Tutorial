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

// Grid / block configuration.
constexpr int THREADS_PER_BLOCK   = 256;
constexpr int NUM_BLOCKS          = 512;   // 512 * 256 = 131072 threads
constexpr int ELEMENTS_PER_THREAD = 256;   // Each thread reads 256 elements

// Total elements in the input array.
constexpr int NUM_THREADS_TOTAL =
    THREADS_PER_BLOCK * NUM_BLOCKS;
constexpr int NUM_ELEMENTS =
    NUM_THREADS_TOTAL * ELEMENTS_PER_THREAD;

// ---------------------------------------------------------------------------
// Coalesced memory access kernel.
//
// Layout interpretation: input is a 2D array of shape
//   [ELEMENTS_PER_THREAD][NUM_THREADS_TOTAL] in row-major order.
// For a fixed "iteration" i, threads in the same warp/wavefront access
//   input[i * NUM_THREADS_TOTAL + tid],
// which are contiguous addresses -> coalesced.
// ---------------------------------------------------------------------------
__global__ void coalesced_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int elements_per_thread,
                                 int num_threads_total)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads_total) return;

    float sum = 0.0f;

    for (int i = 0; i < elements_per_thread; ++i) {
        // Coalesced pattern:
        // For a given i, threads 0..(num_threads_total-1) access
        // input[i * num_threads_total + tid], which is contiguous.
        int idx = i * num_threads_total + tid;
        sum += input[idx];
    }

    output[tid] = sum;
}

int main() {
    std::cout << "Running coalesced_kernel\n";
    std::cout << "  NUM_THREADS_TOTAL   = " << NUM_THREADS_TOTAL   << "\n";
    std::cout << "  ELEMENTS_PER_THREAD = " << ELEMENTS_PER_THREAD << "\n";
    std::cout << "  NUM_ELEMENTS        = " << NUM_ELEMENTS        << "\n";

    // Host buffers.
    std::vector<float> h_input(NUM_ELEMENTS, 1.0f);
    std::vector<float> h_output(NUM_THREADS_TOTAL, 0.0f);

    // Device buffers.
    float *d_input = nullptr, *d_output = nullptr;
    HIP_CHECK(hipMalloc(&d_input,  NUM_ELEMENTS      * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, NUM_THREADS_TOTAL * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_input, h_input.data(),
                        NUM_ELEMENTS * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_output, 0, NUM_THREADS_TOTAL * sizeof(float)));

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(NUM_BLOCKS);

    // Warm-up.
    coalesced_kernel<<<grid, block>>>(d_input, d_output,
                                      ELEMENTS_PER_THREAD, NUM_THREADS_TOTAL);
    HIP_CHECK(hipDeviceSynchronize());

    // Timing.
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    const int num_runs = 50;

    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < num_runs; ++i) {
        coalesced_kernel<<<grid, block>>>(
            d_input, d_output,
            ELEMENTS_PER_THREAD, NUM_THREADS_TOTAL
        );
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float total_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / num_runs;

    std::cout << "Average time over " << num_runs
              << " runs: " << avg_ms << " ms\n";

    // Copy back one sample to check correctness.
    HIP_CHECK(hipMemcpy(h_output.data(), d_output,
                        NUM_THREADS_TOTAL * sizeof(float),
                        hipMemcpyDeviceToHost));

    std::cout << "Sample output[0] = " << h_output[0]
              << " (expected " << ELEMENTS_PER_THREAD << ")\n";

    // Cleanup.
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    return 0;
}
