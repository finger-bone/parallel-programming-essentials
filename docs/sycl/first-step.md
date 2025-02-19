---
sidebar_position: 1
---

# First Step into SYCL

This note book provides introduction to parallel programming. Because this book focuses more on the algorithmic aspects of parallel programming, we use SYCL instead of OpenCL or OpenMP because SYCL is a higher level programming model that is easier to understand and use.

## What is SYCL?

SYCL is a C++ programming model that enables code for heterogeneous processors to be written in a "single-source" style using completely standard C++.

Compared with OpenCL, SYCL is a bit higher level but very easy to use.

SYCL could use OpenCL, CUDA or other backend.

## Run the First SYCL Program

You need to install the oneAPI toolkit. You can download it from the [Intel website](https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit.html). Or alternatively, use AdaptiveCPP compiler. 

<details>
<summary>Docker Environment for Mac Users and Lazy People</summary>

If you are using a Mac like me, unfortunately, the oneAPI DPC++ dropped support for MacOS a while ago. Of course, you can compile and run AdaptiveCpp. But we tool kit from Intel provides more features.

Of course, you can run linux virtual machine. But an easier approach would be to use docker.

```yaml
services:
  hpc:
    image: intel/hpckit
    platform: linux/amd64
    volumes:
      - ./code:/code
    stdin_open: true
    tty: true
    command: /bin/bash
```

Then you can attach vscode to the docker and use it as your environment (Yes, Rosetta is that magical).
</details>

We write our first program as a demonstration. You don't need to understand any of the code. Just copy paste and run it.

```cpp
#include <iostream>
#include <sycl/sycl.hpp>

class vector_addition;

int main(int, char**) {
    sycl::float4 a = { 1.0, 2.0, 3.0, 4.0 };
    sycl::float4 b = { 4.0, 3.0, 2.0, 1.0 };
    sycl::float4 c = { 0.0, 0.0, 0.0, 0.0 };

    auto device_selector = sycl::default_selector_v;

    sycl::queue queue(device_selector);
    
    std::cout << "Running on: "
              << queue.get_device().get_info<sycl::info::device::name>()
              << "\n";

    {
        sycl::buffer<sycl::float4, 1> a_sycl(&a, sycl::range<1>(1));
        sycl::buffer<sycl::float4, 1> b_sycl(&b, sycl::range<1>(1));
        sycl::buffer<sycl::float4, 1> c_sycl(&c, sycl::range<1>(1));

        queue.submit([&] (sycl::handler& cgh) {
            auto a_acc = a_sycl.get_access<sycl::access::mode::read>(cgh);
            auto b_acc = b_sycl.get_access<sycl::access::mode::read>(cgh);
            auto c_acc = c_sycl.get_access<sycl::access::mode::discard_write>(cgh);

            cgh.single_task<class vector_addition>([=] () {
                c_acc[0] = a_acc[0] + b_acc[0];
            });
        });
    }
    std::cout << "  A { " << a.x() << ", " << a.y() << ", " << a.z() << ", " << a.w() << " }\n"
              << "+ B { " << b.x() << ", " << b.y() << ", " << b.z() << ", " << b.w() << " }\n"
              << "==================\n"
              << "= C { " << c.x() << ", " << c.y() << ", " << c.z() << ", " << c.w() << " }"
              << std::endl;

    return 0;
}
```

Use the command,

```bash
icpx -fsycl ./first_step.cc
```

To compile. `icpx` is the Intel compiler. `-fsycl` is the flag to enable SYCL. The output is `a.out`. Run it with `./a.out`.

You should see the output,

```bash
Running on: VirtualApple @ 2.50GHz
  A { 1, 2, 3, 4 }
+ B { 4, 3, 2, 1 }
==================
= C { 5, 5, 5, 5 }
```

The `Running on: VirtualApple @ 2.50GHz` is the name of the device. It may vary depending on the device you are using.

If you ever learnt OpenCL before, you will find the code soothing- not as scary as OpenCL.

For this part, you need to know how to compile and use the header `sycl/sycl.hpp`.

## Where Code Executes

Unlike OpenCL, SYCL uses a single source. So you should be clear about where your code is running on- by that, we mean, if it is running on your host computer, or the device.

:::note

By the host computer, we mean the, well, physical computer you are using. By the device, we mean a part of your computer that is capable of large parallel computation. Typically, it is GPU. But if, for example, you are using Xeon-level processors, the host and device may be the same.

In the following parts of this note, a device always refer to the device that handles large parallel computation.

:::

The host interacts with the device via command queues. This is a queue on the device that stores commands. The host can submit commands to the queue, and the device will execute them in order.

Again, SYCL is single source, thus you must know where your code is going. It is simple: If you are using SYCL specific API to order the device to do something, then it is device code. Otherwise, it is host code.

```cpp
// host code

device.submit([&] (sycl::handler& cgh) {
    cgh.host_task([=] () {
        // host code
    });
    cgh.parallel_for<class my_kernel>(range, [=] (sycl::id<1> idx) {
        // device code
    });
});

// host code
```

```mermaid
graph LR
    A[SYCL Code] -->|Host Code| B[Executed by the Host]
    A -->|Device Code| D[Command Queue]
    D --> E[Executed By Device]
```

Device code are executed asynchronously from the host code. Thus you need synchronization if you need the result.

Host code and device code are fundamentally different. If you ever learnt OpenCL, you can tell that. Thanks to the job done by SYCL dev team, we can write modern C++ for device code. And albeit it runs, you should not use things like dynamic memory allocation, exceptions, etc. in device code. Because most of the time, the device is something like GPU that doesn't support these features.

:::note

When we debug, we usually choose CPU as the device because debugging on CPU is much easier than debugging on GPU.

:::

To select a device, use the `sycl::default_selector_v` as we did in the first program. It will select the default device. You can also use `sycl::cpu_selector_v` or `sycl::gpu_selector_v` to select CPU or GPU.

```cpp
auto device_selector = sycl::default_selector_v;
sycl::queue queue{device_selector};
```

## Action Graph

`q.submit` does not really submit the command to the device. 