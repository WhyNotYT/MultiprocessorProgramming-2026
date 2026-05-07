#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <omp.h>
#include "lodepng.h"

constexpr int WIN_SIZE = 9;
constexpr int MAX_DISP = 65;
constexpr int CROSSCHECK_THRESHOLD = 8;

struct ImageSize
{
    unsigned int width;
    unsigned int height;
};

using GrayImage = std::vector<std::uint8_t>;
using RGBAImage = std::vector<std::uint8_t>;

void CalcZNCC(const GrayImage &left, const GrayImage &right,
              GrayImage &disp_left, GrayImage &disp_right,
              int width, int height)
{
    const int win_half = WIN_SIZE / 2;
    const float inv_win_area = 1.0f / (WIN_SIZE * WIN_SIZE);

    const std::uint8_t *pLeft = left.data();
    const std::uint8_t *pRight = right.data();

#pragma omp parallel for schedule(dynamic, 4) default(none)     \
    shared(pLeft, pRight, disp_left, disp_right, width, height) \
    firstprivate(win_half, inv_win_area, MAX_DISP)
    for (int y = win_half; y < height - win_half; ++y)
    {
        for (int x = win_half; x < width - win_half; ++x)
        {

            float sum_l = 0.0f;
            for (int wy = -win_half; wy <= win_half; ++wy)
            {
                const int row = (y + wy) * width;
                for (int wx = -win_half; wx <= win_half; ++wx)
                    sum_l += pLeft[row + (x + wx)];
            }
            float mean_l = sum_l * inv_win_area;

            float sum_sq_l = 0.0f;
            for (int wy = -win_half; wy <= win_half; ++wy)
            {
                const int row = (y + wy) * width;
                for (int wx = -win_half; wx <= win_half; ++wx)
                {
                    float v = pLeft[row + (x + wx)] - mean_l;
                    sum_sq_l += v * v;
                }
            }
            float std_l = std::sqrt(sum_sq_l);

            float sum_r_base = 0.0f;
            for (int wy = -win_half; wy <= win_half; ++wy)
            {
                const int row = (y + wy) * width;
                for (int wx = -win_half; wx <= win_half; ++wx)
                    sum_r_base += pRight[row + (x + wx)];
            }
            float mean_r_base = sum_r_base * inv_win_area;

            float sum_sq_r_base = 0.0f;
            for (int wy = -win_half; wy <= win_half; ++wy)
            {
                const int row = (y + wy) * width;
                for (int wx = -win_half; wx <= win_half; ++wx)
                {
                    float v = pRight[row + (x + wx)] - mean_r_base;
                    sum_sq_r_base += v * v;
                }
            }
            float std_r_base = std::sqrt(sum_sq_r_base);

            float best_score_l = -1.0f;
            int best_disp_l = 0;
            int max_d_l = std::min(MAX_DISP, x - win_half);

            for (int d = 0; d <= max_d_l; ++d)
            {
                float sum_r = 0.0f;
                for (int wy = -win_half; wy <= win_half; ++wy)
                {
                    const int row = (y + wy) * width;
                    for (int wx = -win_half; wx <= win_half; ++wx)
                        sum_r += pRight[row + (x - d + wx)];
                }
                float mean_r = sum_r * inv_win_area;

                float sum_cross = 0.0f, sum_sq_r = 0.0f;
                for (int wy = -win_half; wy <= win_half; ++wy)
                {
                    const int row = (y + wy) * width;
                    for (int wx = -win_half; wx <= win_half; ++wx)
                    {
                        float vl = pLeft[row + (x + wx)] - mean_l;
                        float vr = pRight[row + (x - d + wx)] - mean_r;
                        sum_cross += vl * vr;
                        sum_sq_r += vr * vr;
                    }
                }
                float std_r = std::sqrt(sum_sq_r);
                float den = std_l * std_r;
                float score = (den > 1e-4f) ? (sum_cross / den) : -1.0f;

                if (score > best_score_l)
                {
                    best_score_l = score;
                    best_disp_l = d;
                }
            }
            disp_left[y * width + x] = static_cast<std::uint8_t>(best_disp_l);

            float best_score_r = -1.0f;
            int best_disp_r = 0;
            int max_d_r = std::min(MAX_DISP, width - 1 - (x + win_half));

            for (int d = 0; d <= max_d_r; ++d)
            {
                float sum_l_shift = 0.0f;
                for (int wy = -win_half; wy <= win_half; ++wy)
                {
                    const int row = (y + wy) * width;
                    for (int wx = -win_half; wx <= win_half; ++wx)
                        sum_l_shift += pLeft[row + (x + d + wx)];
                }
                float mean_l_shift = sum_l_shift * inv_win_area;

                float sum_cross_r = 0.0f, sum_sq_l_shift = 0.0f;
                for (int wy = -win_half; wy <= win_half; ++wy)
                {
                    const int row = (y + wy) * width;
                    for (int wx = -win_half; wx <= win_half; ++wx)
                    {
                        float vr = pRight[row + (x + wx)] - mean_r_base;
                        float vl = pLeft[row + (x + d + wx)] - mean_l_shift;
                        sum_cross_r += vr * vl;
                        sum_sq_l_shift += vl * vl;
                    }
                }
                float std_l_shift = std::sqrt(sum_sq_l_shift);
                float den_r = std_r_base * std_l_shift;
                float score_r = (den_r > 1e-4f) ? (sum_cross_r / den_r) : -1.0f;

                if (score_r > best_score_r)
                {
                    best_score_r = score_r;
                    best_disp_r = d;
                }
            }
            disp_right[y * width + x] = static_cast<std::uint8_t>(best_disp_r);
        }
    }
}

void CrossCheck(const GrayImage &disp_left, const GrayImage &disp_right,
                GrayImage &output, int width, int height)
{
#pragma omp parallel for schedule(static) default(none) \
    shared(disp_left, disp_right, output, width, height)
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int dl = disp_left[y * width + x];
            int x_right = x - dl;
            int dr = (x_right >= 0 && x_right < width)
                         ? disp_right[y * width + x_right]
                         : 0;
            output[y * width + x] =
                (std::abs(dl - dr) > CROSSCHECK_THRESHOLD)
                    ? 0
                    : static_cast<std::uint8_t>(dl);
        }
    }
}

void OcclusionFill(const GrayImage &input, GrayImage &output, int width, int height)
{
#pragma omp parallel for schedule(static) default(none) \
    shared(input, output, width, height)
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            if (input[y * width + x] != 0)
            {
                output[y * width + x] = input[y * width + x];
            }
            else
            {
                std::uint8_t fill_val = 0;
                for (int offset = 1; offset < width; ++offset)
                {
                    if (x - offset >= 0 && input[y * width + (x - offset)] != 0)
                    {
                        fill_val = input[y * width + (x - offset)];
                        break;
                    }
                    if (x + offset < width && input[y * width + (x + offset)] != 0)
                    {
                        fill_val = input[y * width + (x + offset)];
                        break;
                    }
                }
                output[y * width + x] = fill_val;
            }
        }
    }
}

void ProcessImage(const RGBAImage &input_rgba, int w, int h, GrayImage &output_gray)
{
    int nw = w / 4, nh = h / 4;

#pragma omp parallel for schedule(static) default(none) \
    shared(input_rgba, output_gray, w, nw, nh)
    for (int y = 0; y < nh; ++y)
    {
        for (int x = 0; x < nw; ++x)
        {
            int idx = ((y * 4 * w) + (x * 4)) * 4;
            float r = input_rgba[idx];
            float g = input_rgba[idx + 1];
            float b = input_rgba[idx + 2];
            output_gray[y * nw + x] =
                static_cast<std::uint8_t>(0.2126f * r + 0.7152f * g + 0.0722f * b);
        }
    }
}

void SaveNormalizedImage(const std::string &filename,
                         const GrayImage &data, int width, int height)
{
    RGBAImage rgba(width * height * 4);
    for (int i = 0; i < width * height; ++i)
    {
        auto val = static_cast<std::uint8_t>((data[i] * 255) / MAX_DISP);
        rgba[i * 4 + 0] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = val;
        rgba[i * 4 + 3] = 255;
    }
    lodepng::encode(filename, rgba, width, height);
}

int main(int argc, char *argv[])
{

    if (argc > 1)
    {
        int n = std::atoi(argv[1]);
        if (n > 0)
            omp_set_num_threads(n);
    }

    std::cout << "Using " << omp_get_max_threads() << " OpenMP thread(s)." << std::endl;

    RGBAImage img0_raw, img1_raw;
    unsigned int w, h;

    if (lodepng::decode(img0_raw, w, h, "im0.png") ||
        lodepng::decode(img1_raw, w, h, "im1.png"))
    {
        std::cerr << "Could not load images" << std::endl;
        return 1;
    }

    int nw = static_cast<int>(w) / 4;
    int nh = static_cast<int>(h) / 4;
    size_t new_size = static_cast<size_t>(nw) * nh;

    GrayImage gray0(new_size), gray1(new_size);
    GrayImage d_left(new_size, 0), d_right(new_size, 0);
    GrayImage d_cc(new_size, 0), d_final(new_size, 0);

    std::cout << "Processing " << nw << "×" << nh << " image" << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();

    ProcessImage(img0_raw, w, h, gray0);
    auto t1 = std::chrono::high_resolution_clock::now();

    ProcessImage(img1_raw, w, h, gray1);
    auto t2 = std::chrono::high_resolution_clock::now();

    CalcZNCC(gray0, gray1, d_left, d_right, nw, nh);
    auto t3 = std::chrono::high_resolution_clock::now();

    CrossCheck(d_left, d_right, d_cc, nw, nh);
    auto t4 = std::chrono::high_resolution_clock::now();

    OcclusionFill(d_cc, d_final, nw, nh);
    auto t5 = std::chrono::high_resolution_clock::now();

    auto ms = [](auto a, auto b)
    {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    std::cout << "\nTimings:\n";
    std::cout << "  Resize+Grayscale im0: " << ms(t0, t1) << " ms\n";
    std::cout << "  Resize+Grayscale im1: " << ms(t1, t2) << " ms\n";
    std::cout << "  ZNCC (left+right):    " << ms(t2, t3) << " ms\n";
    std::cout << "  Cross-check:          " << ms(t3, t4) << " ms\n";
    std::cout << "  Occlusion fill:       " << ms(t4, t5) << " ms\n";
    std::cout << "  Total:                " << ms(t0, t5) << " ms\n";

    SaveNormalizedImage("depthmap_parallel.png", d_final, nw, nh);
    std::cout << "Result saved" << std::endl;

    return 0;
}