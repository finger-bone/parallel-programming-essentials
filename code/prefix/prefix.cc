#include <array>
#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
    queue q;
    std::cout << "Selected device: "
    << q.get_device().get_info<info::device::name>() << "\n";
    constexpr int N = 1 << 28;
    std::vector<int> d(N, 1);
    buffer<int> buf_prev(
        d.data(), range{N}
    );
    buffer<int> buf_next{range{N}};

    int chunk_size = 1;
    while(chunk_size < N) {
        auto e = q.submit([&](handler& h) {
            accessor prev{buf_prev, h, read_only};
            accessor next{buf_next, h, write_only};

            h.parallel_for(range{N}, [=](id<1> i) {
                int prev_chunk_id = i / chunk_size;
                if (prev_chunk_id % 2 == 0) {
                    next[i] = prev[i];  // Copy the value directly
                    return;
                }
                int prefix_sum_id = prev_chunk_id * chunk_size - 1;
                if (prefix_sum_id >= 0) {  // Ensure that prefix_sum_id is valid
                    next[i] = prev[i] + prev[prefix_sum_id];
                } else {
                    next[i] = prev[i];
                }
            });
        });
        e.wait();

        // Swap buffers using a copy operation
        auto mv = q.submit([&](handler& h) {
            accessor prev{buf_prev, h, write_only};
            accessor next{buf_next, h, read_only};
            h.copy(next, prev);  // Copy back the results to the original buffer
        });
        mv.wait();

        chunk_size <<= 1;  // Double the chunk size
    }
    host_accessor h_access{buf_prev};

    std::cout << h_access[N - 1] << '\n';

    return 0;
}
