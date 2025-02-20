#include <vector>
#include <array>
#include <iostream>
#include <algorithm>
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
    queue q;
    std::cout << "Selected device: "
              << q.get_device().get_info<info::device::name>() << "\n";
    constexpr int N = 1 << 28;
    std::vector<int> d(N, 0);
    buffer prev{d.data(), range{N}};
    buffer<int> next{range{N}};

    auto gen = q.submit([&](handler& h) {
        accessor prev_acc{prev, h, write_only};
        h.parallel_for(N, [=](id<1> i) {
            prev_acc[i] = N - i * (
                1 + (i % 3 == 0) + (i % 5 == 0) + (i % 7 == 0) + (i % 11 == 0)
            );
        });
    });
    gen.wait();


    int chunk_size = 1;
    while (chunk_size < N) {
        auto merge = q.submit([&](handler& h) {
            accessor prev_acc{prev, h, read_only};
            accessor next_acc{next, h, write_only};
            h.parallel_for(N / (2 * chunk_size), [=](id<1> i) {
                int left_start = i * 2 * chunk_size;
                int right_start = left_start + chunk_size;
                int left_end = right_start - 1;
                int right_end = right_start + chunk_size - 1;
                int l = left_start;
                int r = right_start;
                int cur = left_start;
                while (l <= left_end || r <= right_end) {
                    if (l <= left_end && (r > right_end || prev_acc[l] <= prev_acc[r])) {
                        next_acc[cur] = prev_acc[l];
                        ++l;
                    } else {
                        next_acc[cur] = prev_acc[r];
                        ++r;
                    }
                    ++cur;
                }
            });
        });
        merge.wait();
        auto mv = q.submit([&](handler& h) {
            accessor prev_acc{prev, h};
            accessor next_acc{next, h};
            h.copy(next_acc, prev_acc);
        });
        mv.wait();

        chunk_size <<= 1;
    }

    host_accessor h_acc{prev, read_only};
    std::cout << h_acc[N - 1] << std::endl;
}