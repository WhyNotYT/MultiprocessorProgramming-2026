// GPU stereo depth map using OpenCL
// Offloads all processing to GPU: resize, grayscale, ZNCC, cross-check, occlusion fill
// Picks NVIDIA GPU by default, set USE_IGPU=1 to use integrated GPU instead

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <stdexcept>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "lodepng.h"

constexpr int WIN_SIZE = 9;
constexpr int MAX_DISP = 65;
constexpr int CROSSCHECK_THRESH = 8;

// helper macro to check OpenCL errors and exit on failure
#define CL_CHECK(err, msg)                            \
    do                                                \
    {                                                 \
        if ((err) != CL_SUCCESS)                      \
        {                                             \
            std::cerr << "[OpenCL] " << (msg)         \
                      << "  (err=" << (err) << ")\n"; \
            std::exit(1);                             \
        }                                             \
    } while (0)

// read a text file (used to load the kernel source)
static std::string LoadFile(const std::string &path)
{
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("Cannot open: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// compile OpenCL source and print build log on error
static cl_program BuildProgram(cl_context ctx, cl_device_id dev,
                               const std::string &src)
{
    cl_int err;
    const char *csrc = src.c_str();
    size_t slen = src.size();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &csrc, &slen, &err);
    CL_CHECK(err, "clCreateProgramWithSource");

    err = clBuildProgram(prog, 1, &dev, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        size_t log_size = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Build log:\n"
                  << log << "\n";
        std::exit(1);
    }
    return prog;
}

// scale disparity to 0-255 and save as PNG
static void SaveNormalized(const std::string &filename,
                           const std::vector<uint8_t> &data,
                           int width, int height)
{
    std::vector<uint8_t> rgba(width * height * 4);
    for (int i = 0; i < width * height; ++i)
    {
        uint8_t v = static_cast<uint8_t>((data[i] * 255) / MAX_DISP);
        rgba[i * 4 + 0] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = v;
        rgba[i * 4 + 3] = 255;
    }
    lodepng::encode(filename, rgba, width, height);
}

int main(int argc, char *argv[])
{

    std::vector<uint8_t> img0_raw, img1_raw;
    unsigned int W, H;
    if (lodepng::decode(img0_raw, W, H, "im0.png") ||
        lodepng::decode(img1_raw, W, H, "im1.png"))
    {
        std::cerr << "Failed to load images\n";
        return 1;
    }
    const int nw = static_cast<int>(W) / 4;
    const int nh = static_cast<int>(H) / 4;
    const size_t small_px = static_cast<size_t>(nw) * nh;

    std::cout << "Input: " << W << "×" << H
              << "  ->  working at " << nw << "×" << nh << "\n";

    cl_int err;

    // enumerate all platforms and find the right GPU
    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    const char *use_igpu_env = std::getenv("USE_IGPU");
    bool want_igpu = (use_igpu_env != nullptr && std::string(use_igpu_env) == "1");

    cl_device_id device = nullptr;
    for (auto &plat : platforms)
    {
        cl_uint nd = 0;
        if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) == CL_SUCCESS && nd > 0)
        {
            std::vector<cl_device_id> devs(nd);
            clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, nd, devs.data(), nullptr);

            for (auto &d : devs)
            {
                char dname[256];
                clGetDeviceInfo(d, CL_DEVICE_NAME, 256, dname, nullptr);
                std::string name_str(dname);
                bool is_nvidia = (name_str.find("NVIDIA") != std::string::npos);

                if (want_igpu && !is_nvidia)
                {
                    device = d;
                    break;
                }
                else if (!want_igpu && is_nvidia)
                {
                    device = d;
                    break;
                }
            }
        }
        if (device)
            break;
    }

    // Fallback: If we didn't find the preferred one, just grab the first available GPU
    if (!device && !platforms.empty())
    {
        clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    }

    char name[256] = {};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    std::cout << "Using: " << name << "\n";

    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CL_CHECK(err, "clCreateContext");

    // profiling enabled so we can measure per-kernel times
    cl_command_queue queue = clCreateCommandQueue(ctx, device,
                                                  CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err, "clCreateCommandQueue");

    // load and compile all kernels from file
    std::string src = LoadFile("kernels.old.cl");
    cl_program prog = BuildProgram(ctx, device, src);

    cl_kernel k_resize = clCreateKernel(prog, "resize_image", &err);
    CL_CHECK(err, "resize_image");
    cl_kernel k_gray = clCreateKernel(prog, "convert_grayscale", &err);
    CL_CHECK(err, "convert_grayscale");
    cl_kernel k_zncc = clCreateKernel(prog, "zncc", &err);
    CL_CHECK(err, "zncc");
    cl_kernel k_cc = clCreateKernel(prog, "cross_check", &err);
    CL_CHECK(err, "cross_check");
    cl_kernel k_fill = clCreateKernel(prog, "occlusion_fill", &err);
    CL_CHECK(err, "occlusion_fill");

    auto start_transfer_in = std::chrono::high_resolution_clock::now();

    // upload full-res images to GPU as 2D image objects (enables hardware sampling)
    cl_image_format fmt_rgba = {CL_RGBA, CL_UNSIGNED_INT8};
    cl_image_desc desc_full = {CL_MEM_OBJECT_IMAGE2D, W, H};

    cl_mem img0_full = clCreateImage(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     &fmt_rgba, &desc_full, img0_raw.data(), &err);
    cl_mem img1_full = clCreateImage(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     &fmt_rgba, &desc_full, img1_raw.data(), &err);

    clFinish(queue);
    auto end_transfer_in = std::chrono::high_resolution_clock::now();

    // allocate small (downscaled) images and intermediate buffers on GPU
    cl_image_desc desc_small = {CL_MEM_OBJECT_IMAGE2D, (size_t)nw, (size_t)nh};
    cl_mem img0_small = clCreateImage(ctx, CL_MEM_READ_WRITE, &fmt_rgba, &desc_small, nullptr, &err);
    cl_mem img1_small = clCreateImage(ctx, CL_MEM_READ_WRITE, &fmt_rgba, &desc_small, nullptr, &err);
    cl_mem buf_gray0 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, small_px, nullptr, &err);
    cl_mem buf_gray1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, small_px, nullptr, &err);
    cl_mem buf_dl = clCreateBuffer(ctx, CL_MEM_READ_WRITE, small_px, nullptr, &err);
    cl_mem buf_dr = clCreateBuffer(ctx, CL_MEM_READ_WRITE, small_px, nullptr, &err);
    cl_mem buf_cc = clCreateBuffer(ctx, CL_MEM_READ_WRITE, small_px, nullptr, &err);
    cl_mem buf_out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, small_px, nullptr, &err);

    // initialize disparity buffers to zero
    uint8_t zero = 0;
    clEnqueueFillBuffer(queue, buf_dl, &zero, sizeof(uint8_t), 0, small_px, 0, nullptr, nullptr);
    clEnqueueFillBuffer(queue, buf_dr, &zero, sizeof(uint8_t), 0, small_px, 0, nullptr, nullptr);

    auto start_device = std::chrono::high_resolution_clock::now();

    size_t gs2d[2] = {(size_t)nw, (size_t)nh};
    cl_event ev[7] = {};

    // resize both images
    clSetKernelArg(k_resize, 0, sizeof(cl_mem), &img0_full);
    clSetKernelArg(k_resize, 1, sizeof(cl_mem), &img0_small);
    clEnqueueNDRangeKernel(queue, k_resize, 2, nullptr, gs2d, nullptr, 0, nullptr, &ev[0]);
    clSetKernelArg(k_resize, 0, sizeof(cl_mem), &img1_full);
    clSetKernelArg(k_resize, 1, sizeof(cl_mem), &img1_small);
    clEnqueueNDRangeKernel(queue, k_resize, 2, nullptr, gs2d, nullptr, 0, nullptr, &ev[1]);

    // convert both to grayscale
    clSetKernelArg(k_gray, 0, sizeof(cl_mem), &img0_small);
    clSetKernelArg(k_gray, 1, sizeof(cl_mem), &buf_gray0);
    clSetKernelArg(k_gray, 2, sizeof(int), &nw);
    clEnqueueNDRangeKernel(queue, k_gray, 2, nullptr, gs2d, nullptr, 0, nullptr, &ev[2]);
    clSetKernelArg(k_gray, 0, sizeof(cl_mem), &img1_small);
    clSetKernelArg(k_gray, 1, sizeof(cl_mem), &buf_gray1);
    clSetKernelArg(k_gray, 2, sizeof(int), &nw);
    clEnqueueNDRangeKernel(queue, k_gray, 2, nullptr, gs2d, nullptr, 0, nullptr, &ev[3]);

    // ZNCC: compute both left and right disparity maps in one kernel
    const int win_half = WIN_SIZE / 2;
    const int max_disp = MAX_DISP;
    clSetKernelArg(k_zncc, 0, sizeof(cl_mem), &buf_gray0);
    clSetKernelArg(k_zncc, 1, sizeof(cl_mem), &buf_gray1);
    clSetKernelArg(k_zncc, 2, sizeof(cl_mem), &buf_dl);
    clSetKernelArg(k_zncc, 3, sizeof(cl_mem), &buf_dr);
    clSetKernelArg(k_zncc, 4, sizeof(int), &nw);
    clSetKernelArg(k_zncc, 5, sizeof(int), &nh);
    clSetKernelArg(k_zncc, 6, sizeof(int), &win_half);
    clSetKernelArg(k_zncc, 7, sizeof(int), &max_disp);
    clEnqueueNDRangeKernel(queue, k_zncc, 2, nullptr, gs2d, nullptr, 0, nullptr, &ev[4]);

    // cross-check: zero out inconsistent disparities
    const int cc_thresh = CROSSCHECK_THRESH;
    clSetKernelArg(k_cc, 0, sizeof(cl_mem), &buf_dl);
    clSetKernelArg(k_cc, 1, sizeof(cl_mem), &buf_dr);
    clSetKernelArg(k_cc, 2, sizeof(cl_mem), &buf_cc);
    clSetKernelArg(k_cc, 3, sizeof(int), &nw);
    clSetKernelArg(k_cc, 4, sizeof(int), &nh);
    clSetKernelArg(k_cc, 5, sizeof(int), &cc_thresh);
    clEnqueueNDRangeKernel(queue, k_cc, 2, nullptr, gs2d, nullptr, 0, nullptr, &ev[5]);

    // occlusion fill: one thread per row to fill holes
    size_t gs1d[1] = {(size_t)nh};
    clSetKernelArg(k_fill, 0, sizeof(cl_mem), &buf_cc);
    clSetKernelArg(k_fill, 1, sizeof(cl_mem), &buf_out);
    clSetKernelArg(k_fill, 2, sizeof(int), &nw);
    clSetKernelArg(k_fill, 3, sizeof(int), &nh);
    clEnqueueNDRangeKernel(queue, k_fill, 1, nullptr, gs1d, nullptr, 0, nullptr, &ev[6]);

    CL_CHECK(clFinish(queue), "clFinish Execution");
    auto end_device = std::chrono::high_resolution_clock::now();

    auto start_transfer_out = std::chrono::high_resolution_clock::now();

    // read result back to CPU
    std::vector<uint8_t> result(small_px);
    CL_CHECK(clEnqueueReadBuffer(queue, buf_out, CL_TRUE, 0, small_px, result.data(), 0, nullptr, nullptr), "read result");

    auto end_transfer_out = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> d_in = end_transfer_in - start_transfer_in;
    std::chrono::duration<double, std::milli> d_dev = end_device - start_device;
    std::chrono::duration<double, std::milli> d_out = end_transfer_out - start_transfer_out;

    static const char *knames[] = {
        "  Resize im0         ",
        "  Resize im1         ",
        "  Grayscale im0      ",
        "  Grayscale im1      ",
        "  ZNCC (left+right)  ",
        "  Cross-check        ",
        "  Occlusion fill     ",
    };

    // print per-kernel timing using OpenCL profiling events
    std::cout << "\nTimings:\n";
    std::cout << "  CPU -> GPU transfer: " << d_in.count() << " ms\n";
    for (int i = 0; i < 7; ++i)
    {
        cl_ulong ts = 0, te = 0;
        clGetEventProfilingInfo(ev[i], CL_PROFILING_COMMAND_START, sizeof(ts), &ts, nullptr);
        clGetEventProfilingInfo(ev[i], CL_PROFILING_COMMAND_END, sizeof(te), &te, nullptr);
        std::cout << knames[i] << ": " << (te - ts) * 1e-6 << " ms\n";
        clReleaseEvent(ev[i]);
    }
    std::cout << "  GPU -> CPU transfer: " << d_out.count() << " ms\n";
    std::cout << "  GPU kernels total:   " << d_dev.count() << " ms\n";
    std::cout << "  Total:               " << (d_in.count() + d_dev.count() + d_out.count()) << " ms\n";

    SaveNormalized("depthmap_opencl.png", result, nw, nh);
    std::cout << "Result saved" << std::endl;

    // cleanup
    clReleaseMemObject(img0_full);
    clReleaseMemObject(img1_full);
    clReleaseMemObject(img0_small);
    clReleaseMemObject(img1_small);
    clReleaseMemObject(buf_gray0);
    clReleaseMemObject(buf_gray1);
    clReleaseMemObject(buf_dl);
    clReleaseMemObject(buf_dr);
    clReleaseMemObject(buf_cc);
    clReleaseMemObject(buf_out);
    clReleaseKernel(k_resize);
    clReleaseKernel(k_gray);
    clReleaseKernel(k_zncc);
    clReleaseKernel(k_cc);
    clReleaseKernel(k_fill);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return 0;
}