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
constexpr int WG_X = 32;
constexpr int WG_Y = 8;
/* OPT-A: fill passes use 1-D workgroups of this size (rows per WG) */
constexpr int WG_FILL = 64;

static const char *BUILD_FLAGS =
    "-cl-fast-relaxed-math -cl-mad-enable -cl-no-signed-zeros";

static void cl_die(cl_int e, const char *msg, int line)
{
    std::cerr << "FATAL line " << line << ": " << msg
              << "  (err=" << e << ")\n";
    std::exit(1);
}
#define CK(call, msg)                  \
    do                                 \
    {                                  \
        cl_int _e = (call);            \
        if (_e != CL_SUCCESS)          \
            cl_die(_e, msg, __LINE__); \
    } while (0)

#define STEP(s)                             \
    do                                      \
    {                                       \
        std::cout << "  >> " << (s) << "\n" \
                  << std::flush;            \
    } while (0)

static std::string LoadFile(const std::string &path)
{
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("Cannot open: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static cl_program BuildProgram(cl_context ctx, cl_device_id dev,
                               const std::string &src)
{
    cl_int err;
    const char *csrc = src.c_str();
    size_t slen = src.size();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &csrc, &slen, &err);
    CK(err, "clCreateProgramWithSource");
    err = clBuildProgram(prog, 1, &dev, BUILD_FLAGS, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        size_t sz = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &sz);
        std::string log(sz, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sz, log.data(), nullptr);
        std::cerr << "Build log:\n"
                  << log << "\n";
        std::exit(1);
    }
    return prog;
}

static void SaveNormalized(const std::string &fn,
                           const std::vector<uint8_t> &data, int w, int h)
{
    std::vector<uint8_t> rgba(w * h * 4);
    for (int i = 0; i < w * h; ++i)
    {
        uint8_t v = (uint8_t)((data[i] * 255) / MAX_DISP);
        rgba[i * 4] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = v;
        rgba[i * 4 + 3] = 255;
    }
    lodepng::encode(fn, rgba, w, h);
}

static size_t RoundUp(size_t n, size_t m) { return ((n + m - 1) / m) * m; }

int main(int, char **)
{
    STEP("loading images");
    std::vector<uint8_t> img0_raw, img1_raw;
    unsigned int W, H;
    if (lodepng::decode(img0_raw, W, H, "im0.png") ||
        lodepng::decode(img1_raw, W, H, "im1.png"))
    {
        std::cerr << "Failed to load images\n";
        return 1;
    }

    const int nw = (int)W / 4, nh = (int)H / 4;
    const size_t small_px = (size_t)nw * nh;
    std::cout << "Input: " << W << "x" << H
              << " -> " << nw << "x" << nh << "\n"
              << std::flush;

    STEP("selecting device");
    cl_int err;
    cl_uint np = 0;

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

    char dname[256] = {};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dname), dname, nullptr);
    std::cout << "GPU: " << dname << "\n"
              << std::flush;

    STEP("creating context");
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CK(err, "clCreateContext");

    STEP("creating in-order queue");
    cl_command_queue queue = clCreateCommandQueue(ctx, device,
                                                  CL_QUEUE_PROFILING_ENABLE, &err);
    CK(err, "clCreateCommandQueue");

    STEP("building kernels");
    std::string src = LoadFile("kernels.optimized.cl");
    cl_program prog = BuildProgram(ctx, device, src);

    /* FIX-3 / OPT-A: one object per kernel function.
     * fill_pass1 + fill_pass2 replace the old single occlusion_fill kernel. */
    STEP("creating kernel objects");
    cl_kernel k_resize = clCreateKernel(prog, "resize_image", &err);
    CK(err, "k_resize");
    cl_kernel k_gray = clCreateKernel(prog, "convert_grayscale", &err);
    CK(err, "k_gray");
    cl_kernel k_znl = clCreateKernel(prog, "zncc_left", &err);
    CK(err, "k_znl");
    cl_kernel k_znr = clCreateKernel(prog, "zncc_right", &err);
    CK(err, "k_znr");
    cl_kernel k_cc = clCreateKernel(prog, "cross_check", &err);
    CK(err, "k_cc");
    cl_kernel k_fp1 = clCreateKernel(prog, "fill_pass1", &err);
    CK(err, "k_fp1");
    cl_kernel k_fp2 = clCreateKernel(prog, "fill_pass2", &err);
    CK(err, "k_fp2");

    STEP("allocating full-res images on GPU");
    cl_image_format fmt = {CL_RGBA, CL_UNSIGNED_INT8};
    cl_image_desc dfull = {};
    dfull.image_type = CL_MEM_OBJECT_IMAGE2D;
    dfull.image_width = W;
    dfull.image_height = H;
    cl_mem img0f = clCreateImage(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 &fmt, &dfull, img0_raw.data(), &err);
    CK(err, "img0_full");
    cl_mem img1f = clCreateImage(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 &fmt, &dfull, img1_raw.data(), &err);
    CK(err, "img1_full");

    STEP("allocating small images on GPU");
    cl_image_desc dsml = {};
    dsml.image_type = CL_MEM_OBJECT_IMAGE2D;
    dsml.image_width = nw;
    dsml.image_height = nh;
    cl_mem img0s = clCreateImage(ctx, CL_MEM_READ_WRITE, &fmt, &dsml, nullptr, &err);
    CK(err, "img0s");
    cl_mem img1s = clCreateImage(ctx, CL_MEM_READ_WRITE, &fmt, &dsml, nullptr, &err);
    CK(err, "img1s");

    STEP("allocating buffers on GPU");
    cl_mem bgry0 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, small_px, nullptr, &err);
    CK(err, "bgry0");
    cl_mem bgry1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, small_px, nullptr, &err);
    CK(err, "bgry1");
    cl_mem bdl = clCreateBuffer(ctx, CL_MEM_READ_WRITE, small_px, nullptr, &err);
    CK(err, "bdl");
    cl_mem bdr = clCreateBuffer(ctx, CL_MEM_READ_WRITE, small_px, nullptr, &err);
    CK(err, "bdr");
    cl_mem bcc = clCreateBuffer(ctx, CL_MEM_READ_WRITE, small_px, nullptr, &err);
    CK(err, "bcc");
    /* OPT-A: scratch buffer for fill pass 1 → pass 2 hand-off.
     * 2 bytes per pixel: [fill_value, distance_to_source].               */
    cl_mem bscratch = clCreateBuffer(ctx, CL_MEM_READ_WRITE, small_px * 2, nullptr, &err);
    CK(err, "bscratch");
    cl_mem bout = clCreateBuffer(ctx, CL_MEM_READ_WRITE, small_px, nullptr, &err);
    CK(err, "bout");

    /* NDRange sizes */
    size_t gs2[2] = {RoundUp(nw, WG_X), RoundUp(nh, WG_Y)}, ls2[2] = {WG_X, WG_Y};
    size_t gs1[1] = {RoundUp(nh, WG_FILL)}, ls1[1] = {WG_FILL};

    /* 10 events: resize×2, gray×2, zncc×2, cross_check, fill_pass1, fill_pass2 */
    cl_event ev[9] = {};
    auto t_upload_start = std::chrono::high_resolution_clock::now();
    clFinish(queue); /* wait for image uploads triggered by CL_MEM_COPY_HOST_PTR */
    auto t0 = std::chrono::high_resolution_clock::now();

    /* ---- resize img0 ---- */
    STEP("enqueue resize img0");
    CK(clSetKernelArg(k_resize, 0, sizeof(cl_mem), &img0f), "resize a0");
    CK(clSetKernelArg(k_resize, 1, sizeof(cl_mem), &img0s), "resize a1");
    CK(clEnqueueNDRangeKernel(queue, k_resize, 2, nullptr, gs2, ls2, 0, nullptr, &ev[0]), "enq res0");

    /* ---- resize img1 – reuse k_resize (FIX-3) ---- */
    STEP("enqueue resize img1");
    CK(clSetKernelArg(k_resize, 0, sizeof(cl_mem), &img1f), "resize a0");
    CK(clSetKernelArg(k_resize, 1, sizeof(cl_mem), &img1s), "resize a1");
    CK(clEnqueueNDRangeKernel(queue, k_resize, 2, nullptr, gs2, ls2, 0, nullptr, &ev[1]), "enq res1");

    /* ---- grayscale img0 ---- */
    STEP("enqueue grayscale img0");
    CK(clSetKernelArg(k_gray, 0, sizeof(cl_mem), &img0s), "gray a0");
    CK(clSetKernelArg(k_gray, 1, sizeof(cl_mem), &bgry0), "gray a1");
    CK(clSetKernelArg(k_gray, 2, sizeof(int), &nw), "gray a2");
    CK(clSetKernelArg(k_gray, 3, sizeof(int), &nh), "gray a3");
    CK(clEnqueueNDRangeKernel(queue, k_gray, 2, nullptr, gs2, ls2, 0, nullptr, &ev[2]), "enq gry0");

    /* ---- grayscale img1 – reuse k_gray (FIX-3) ---- */
    STEP("enqueue grayscale img1");
    CK(clSetKernelArg(k_gray, 0, sizeof(cl_mem), &img1s), "gray a0");
    CK(clSetKernelArg(k_gray, 1, sizeof(cl_mem), &bgry1), "gray a1");
    CK(clSetKernelArg(k_gray, 2, sizeof(int), &nw), "gray a2");
    CK(clSetKernelArg(k_gray, 3, sizeof(int), &nh), "gray a3");
    CK(clEnqueueNDRangeKernel(queue, k_gray, 2, nullptr, gs2, ls2, 0, nullptr, &ev[3]), "enq gry1");

    /* ---- zncc_left ---- */
    STEP("enqueue zncc_left");
    const int win_half = WIN_SIZE / 2, max_disp = MAX_DISP;
    CK(clSetKernelArg(k_znl, 0, sizeof(cl_mem), &bgry0), "znl a0");
    CK(clSetKernelArg(k_znl, 1, sizeof(cl_mem), &bgry1), "znl a1");
    CK(clSetKernelArg(k_znl, 2, sizeof(cl_mem), &bdl), "znl a2");
    CK(clSetKernelArg(k_znl, 3, sizeof(int), &nw), "znl a3");
    CK(clSetKernelArg(k_znl, 4, sizeof(int), &nh), "znl a4");
    CK(clSetKernelArg(k_znl, 5, sizeof(int), &win_half), "znl a5");
    CK(clSetKernelArg(k_znl, 6, sizeof(int), &max_disp), "znl a6");
    CK(clEnqueueNDRangeKernel(queue, k_znl, 2, nullptr, gs2, ls2, 0, nullptr, &ev[4]), "enq znl");

    /* ---- zncc_right ---- */
    STEP("enqueue zncc_right");
    CK(clSetKernelArg(k_znr, 0, sizeof(cl_mem), &bgry0), "znr a0");
    CK(clSetKernelArg(k_znr, 1, sizeof(cl_mem), &bgry1), "znr a1");
    CK(clSetKernelArg(k_znr, 2, sizeof(cl_mem), &bdr), "znr a2");
    CK(clSetKernelArg(k_znr, 3, sizeof(int), &nw), "znr a3");
    CK(clSetKernelArg(k_znr, 4, sizeof(int), &nh), "znr a4");
    CK(clSetKernelArg(k_znr, 5, sizeof(int), &win_half), "znr a5");
    CK(clSetKernelArg(k_znr, 6, sizeof(int), &max_disp), "znr a6");
    CK(clEnqueueNDRangeKernel(queue, k_znr, 2, nullptr, gs2, ls2, 0, nullptr, &ev[5]), "enq znr");

    /* ---- cross_check ---- */
    STEP("enqueue cross_check");
    const int cc_thresh = CROSSCHECK_THRESH;
    CK(clSetKernelArg(k_cc, 0, sizeof(cl_mem), &bdl), "cc a0");
    CK(clSetKernelArg(k_cc, 1, sizeof(cl_mem), &bdr), "cc a1");
    CK(clSetKernelArg(k_cc, 2, sizeof(cl_mem), &bcc), "cc a2");
    CK(clSetKernelArg(k_cc, 3, sizeof(int), &nw), "cc a3");
    CK(clSetKernelArg(k_cc, 4, sizeof(int), &nh), "cc a4");
    CK(clSetKernelArg(k_cc, 5, sizeof(int), &cc_thresh), "cc a5");
    CK(clEnqueueNDRangeKernel(queue, k_cc, 2, nullptr, gs2, ls2, 0, nullptr, &ev[6]), "enq cc");

    /* ---- OPT-A: fill_pass1  (left→right, writes scratch) ---- */
    STEP("enqueue fill_pass1");
    CK(clSetKernelArg(k_fp1, 0, sizeof(cl_mem), &bcc), "fp1 a0");
    CK(clSetKernelArg(k_fp1, 1, sizeof(cl_mem), &bscratch), "fp1 a1");
    CK(clSetKernelArg(k_fp1, 2, sizeof(int), &nw), "fp1 a2");
    CK(clSetKernelArg(k_fp1, 3, sizeof(int), &nh), "fp1 a3");
    CK(clEnqueueNDRangeKernel(queue, k_fp1, 1, nullptr, gs1, ls1, 0, nullptr, &ev[7]), "enq fp1");

    /* ---- OPT-A: fill_pass2  (right→left, writes bout) ---- */
    STEP("enqueue fill_pass2");
    CK(clSetKernelArg(k_fp2, 0, sizeof(cl_mem), &bcc), "fp2 a0");
    CK(clSetKernelArg(k_fp2, 1, sizeof(cl_mem), &bscratch), "fp2 a1");
    CK(clSetKernelArg(k_fp2, 2, sizeof(cl_mem), &bout), "fp2 a2");
    CK(clSetKernelArg(k_fp2, 3, sizeof(int), &nw), "fp2 a3");
    CK(clSetKernelArg(k_fp2, 4, sizeof(int), &nh), "fp2 a4");
    CK(clEnqueueNDRangeKernel(queue, k_fp2, 1, nullptr, gs1, ls1, 0, nullptr, &ev[8]), "enq fp2");

    STEP("clFinish");
    CK(clFinish(queue), "clFinish");

    auto t1 = std::chrono::high_resolution_clock::now();

    static const char *knames[] = {
        "  Resize im0         ",
        "  Resize im1         ",
        "  Grayscale im0      ",
        "  Grayscale im1      ",
        "  ZNCC left          ",
        "  ZNCC right         ",
        "  Cross-check        ",
        "  Fill pass 1        ",
        "  Fill pass 2        "};
    std::cout << "\nTimings:\n";
    std::cout << "  CPU -> GPU transfer: "
              << std::chrono::duration<double, std::milli>(t0 - t_upload_start).count()
              << " ms\n";
    double kernel_total = 0.0;
    for (int i = 0; i < 9; i++)
    {
        cl_ulong ts = 0, te = 0;
        clGetEventProfilingInfo(ev[i], CL_PROFILING_COMMAND_START, sizeof(ts), &ts, nullptr);
        clGetEventProfilingInfo(ev[i], CL_PROFILING_COMMAND_END, sizeof(te), &te, nullptr);
        double kms = (te - ts) * 1e-6;
        kernel_total += kms;
        std::cout << knames[i] << ": " << kms << " ms\n";
        clReleaseEvent(ev[i]);
    }

    STEP("reading result");
    std::vector<uint8_t> result(small_px);
    CK(clEnqueueReadBuffer(queue, bout, CL_TRUE, 0, small_px,
                           result.data(), 0, nullptr, nullptr),
       "readback");

    auto t2 = std::chrono::high_resolution_clock::now();
    double readback_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    std::cout << "  GPU -> CPU transfer: " << readback_ms << " ms\n";
    std::cout << "  GPU kernels total:   " << kernel_total << " ms\n";
    std::cout << "  Total:               "
              << std::chrono::duration<double, std::milli>(t2 - t_upload_start).count()
              << " ms\n";

    SaveNormalized("depthmap_opencl.png", result, nw, nh);
    std::cout << "Result saved\n";
    clReleaseMemObject(img0f);
    clReleaseMemObject(img1f);
    clReleaseMemObject(img0s);
    clReleaseMemObject(img1s);
    clReleaseMemObject(bgry0);
    clReleaseMemObject(bgry1);
    clReleaseMemObject(bdl);
    clReleaseMemObject(bdr);
    clReleaseMemObject(bcc);
    clReleaseMemObject(bscratch);
    clReleaseMemObject(bout);
    clReleaseKernel(k_resize);
    clReleaseKernel(k_gray);
    clReleaseKernel(k_znl);
    clReleaseKernel(k_znr);
    clReleaseKernel(k_cc);
    clReleaseKernel(k_fp1);
    clReleaseKernel(k_fp2);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}