#include <vector>
#include <array>
#include <iostream>
#include <algorithm>
#include <sycl/sycl.hpp>

using namespace sycl;

int main()
{
    queue q;
    std::cout << "Selected device: "
              << q.get_device().get_info<info::device::name>() << "\n";
    constexpr int N = 1 << 28;
    std::vector<int> d(N, 0);
    buffer prev{d.data(), range{N}};
    buffer<int> next{range{N}};

    auto gen = q.submit([&](handler &h)
                        {
        accessor prev_acc{prev, h, write_only};
        h.parallel_for(N, [=](id<1> i) {
            prev_acc[i] = N - i * (
                1 + (i % 3 == 0) + (i % 5 == 0) + (i % 7 == 0) + (i % 11 == 0)
            );
        }); });
    gen.wait();

    int chunk_size = 1;
    while (chunk_size < N)
    {
        chunk_size <<= 1;
        for (int j = chunk_size >> 1; j > 0; j >>= 1)
        {
            auto merge = q.submit([&](handler &h)
                                  {
            accessor prev_acc{prev, h, read_only};
            accessor next_acc{next, h, write_only};
            int s = chunk_size;
            int step = j;
            
            h.parallel_for(N, [=](id<1> idx) {
                int i = idx[0];
                int k = i / s;
                bool asc = (k % 2 == 0);
                int partner = i ^ step;

                if (partner > i) {
                    if (i & step) return;
                    
                    if ((asc && prev_acc[i] > prev_acc[partner]) || 
                        (!asc && prev_acc[i] < prev_acc[partner])) {
                        next_acc[i] = prev_acc[partner];
                        next_acc[partner] = prev_acc[i];
                    } else {
                        next_acc[i] = prev_acc[i];
                        next_acc[partner] = prev_acc[partner];
                    }
                }
            }); });
            merge.wait();
            auto mv = q.submit([&](handler &h)
                               {
            accessor prev_acc{prev, h};
            accessor next_acc{next, h};
            h.copy(next_acc, prev_acc); });
            mv.wait();
        }
    }

    host_accessor h_acc{prev, read_only};
    std::cout << h_acc[N - 1] << std::endl;
}