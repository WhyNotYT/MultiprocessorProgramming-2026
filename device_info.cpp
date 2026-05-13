#include <iostream>
#include <vector>
#include <string>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

int main()
{
    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    std::cout << "Num platforms detected : " << num_platforms << "\n";

    for (cl_uint pi = 0; pi < num_platforms; ++pi)
    {
        std::cout << "\n\n\n";
        char buf[256];
        clGetPlatformInfo(platforms[pi], CL_PLATFORM_VENDOR, sizeof(buf), buf, nullptr);
        std::cout << "Platform vendor        : " << buf << "\n";
        clGetPlatformInfo(platforms[pi], CL_PLATFORM_NAME, sizeof(buf), buf, nullptr);
        std::cout << "Platform name          : " << buf << "\n";
        clGetPlatformInfo(platforms[pi], CL_PLATFORM_PROFILE, sizeof(buf), buf, nullptr);
        std::cout << "Platform profile       : " << buf << "\n";
        clGetPlatformInfo(platforms[pi], CL_PLATFORM_VERSION, sizeof(buf), buf, nullptr);
        std::cout << "Platform version       : " << buf << "\n";

        cl_uint num_devices = 0;
        clGetDeviceIDs(platforms[pi], CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(platforms[pi], CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr);

        std::cout << "\nNum devices detected   : " << num_devices << "\n";

        for (cl_uint di = 0; di < num_devices; ++di)
        {
            cl_device_id dev = devices[di];

            clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(buf), buf, nullptr);
            std::cout << "Device name            : " << buf << "\n";

            clGetDeviceInfo(dev, CL_DEVICE_VERSION, sizeof(buf), buf, nullptr);
            std::cout << "Device hardware version: " << buf << "\n";

            clGetDeviceInfo(dev, CL_DRIVER_VERSION, sizeof(buf), buf, nullptr);
            std::cout << "Device driver version  : " << buf << "\n";

            clGetDeviceInfo(dev, CL_DEVICE_OPENCL_C_VERSION, sizeof(buf), buf, nullptr);
            std::cout << "Device OpenCL_C version: " << buf << "\n";

            cl_uint compute_units = 0;
            clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr);
            std::cout << "Device parallel compute units: " << compute_units << "\n";

            cl_device_local_mem_type local_mem_type;
            clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_mem_type), &local_mem_type, nullptr);
            std::cout << "CL_DEVICE_LOCAL_MEM_TYPE       : "
                      << (local_mem_type == CL_LOCAL ? "CL_LOCAL" : "CL_GLOBAL") << "\n";

            cl_ulong local_mem_size = 0;
            clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, nullptr);
            std::cout << "CL_DEVICE_LOCAL_MEM_SIZE       : " << local_mem_size << " bytes ("
                      << local_mem_size / 1024 << " KB)\n";

            std::cout << "CL_DEVICE_MAX_COMPUTE_UNITS    : " << compute_units << "\n";

            cl_uint clock_freq = 0;
            clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_freq), &clock_freq, nullptr);
            std::cout << "CL_DEVICE_MAX_CLOCK_FREQUENCY  : " << clock_freq << " MHz\n";

            cl_ulong const_buf_size = 0;
            clGetDeviceInfo(dev, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(const_buf_size), &const_buf_size, nullptr);
            std::cout << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: " << const_buf_size << " bytes ("
                      << const_buf_size / 1024 << " KB)\n";

            size_t max_wg_size = 0;
            clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg_size), &max_wg_size, nullptr);
            std::cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE  : " << max_wg_size << "\n";

            cl_uint max_wi_dims = 0;
            clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_wi_dims), &max_wi_dims, nullptr);
            std::vector<size_t> max_wi_sizes(max_wi_dims);
            clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                            max_wi_dims * sizeof(size_t), max_wi_sizes.data(), nullptr);
            std::cout << "CL_DEVICE_MAX_WORK_ITEM_SIZES  : ";
            for (cl_uint d = 0; d < max_wi_dims; ++d)
                std::cout << max_wi_sizes[d] << (d + 1 < max_wi_dims ? " x " : "\n");
        }
    }

    return 0;
}
