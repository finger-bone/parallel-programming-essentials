#include <array>
#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

#define IMPLICIT

constexpr int N = 42;

#ifndef IMPLICIT
int main() {
    queue q;
    std::array<int, N> host_array;
    int *device_array = malloc_device<int>(N, q);
    // Initialize host_array
    for (int i = 0; i < N; i++) {
        host_array[i] = N;
    }

    // Copy host_array to device_array
    q.submit([&](handler &h) {
        h.memcpy(device_array, &host_array[0], N * sizeof(int));
    });

    q.wait();

    // Increment values in device_array
    q.submit([&](handler &h) {
        h.parallel_for(N, [=](id<1> i) {
            device_array[i]++; 
        });
    });

    q.wait();

    // Copy device_array back to host_array
    q.submit([&](handler &h) {
        h.memcpy(&host_array[0], device_array, N * sizeof(int));
    });

    q.wait();
    std::cout << "host_array[0] = " << host_array[0] << std::endl;
    std::cout << "device_array[0] = " << device_array[0] << std::endl;
    // Free device memory
    free(device_array, q);

    return 0;
}
#endif

#ifdef IMPLICIT

int main() {
    queue q;

    int *host_array = malloc_host<int>(N, q);
    int *shared_array = malloc_shared<int>(N, q);

    // Initialize host_array on host
    for (int i = 0; i < N; i++) {
        host_array[i] = i;
    }

    // We will learn how to simplify this example later
    q.submit([&](handler &h) {
        h.parallel_for(N, [=](id<1> i) {
            // Access shared_array and host_array on device
            shared_array[i] = host_array[i] + 1;
        });
    });

    q.wait();

    // Access shared_array on host
    for (int i = 0; i < N; i++) {
        host_array[i] = shared_array[i];
    }

    free(shared_array, q);
    free(host_array, q);
    return 0;
}

#endif