#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstdint>
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

    for (int y = win_half; y < height - win_half; ++y)
    {
        for (int x = win_half; x < width - win_half; ++x)
        {
            // precalcualte window stats
            float sum_l = 0;
            for (int wy = -win_half; wy <= win_half; ++wy)
            {
                const int row = (y + wy) * width;
                for (int wx = -win_half; wx <= win_half; ++wx)
                {
                    sum_l += pLeft[row + (x + wx)];
                }
            }
            float mean_l = sum_l * inv_win_area;

            float sum_sq_diff_l = 0;
            for (int wy = -win_half; wy <= win_half; ++wy)
            {
                const int row = (y + wy) * width;
                for (int wx = -win_half; wx <= win_half; ++wx)
                {
                    float val = static_cast<float>(pLeft[row + (x + wx)]) - mean_l;
                    sum_sq_diff_l += val * val;
                }
            }
            float std_l = std::sqrt(sum_sq_diff_l);

            // left right search
            float best_score_l = -1.0f;
            int best_disp_l = 0;
            int max_d_l = std::min(MAX_DISP, x - win_half);

            for (int d = 0; d <= max_d_l; ++d)
            {
                float sum_r = 0;
                for (int wy = -win_half; wy <= win_half; ++wy)
                {
                    const int row = (y + wy) * width;
                    for (int wx = -win_half; wx <= win_half; ++wx)
                    {
                        sum_r += pRight[row + (x - d + wx)];
                    }
                }
                float mean_r = sum_r * inv_win_area;

                float sum_cross = 0;
                float sum_sq_diff_r = 0;
                for (int wy = -win_half; wy <= win_half; ++wy)
                {
                    const int row = (y + wy) * width;
                    for (int wx = -win_half; wx <= win_half; ++wx)
                    {
                        float val_l = static_cast<float>(pLeft[row + (x + wx)]) - mean_l;
                        float val_r = static_cast<float>(pRight[row + (x - d + wx)]) - mean_r;
                        sum_cross += val_l * val_r;
                        sum_sq_diff_r += val_r * val_r;
                    }
                }

                float std_r = std::sqrt(sum_sq_diff_r);
                float den = std_l * std_r;
                float score = (den > 0.0001f) ? (sum_cross / den) : -1.0f;

                if (score > best_score_l)
                {
                    best_score_l = score;
                    best_disp_l = d;
                }
            }
            disp_left[y * width + x] = static_cast<std::uint8_t>(best_disp_l);

            // precalculate window states
            float sum_r_base = 0;
            for (int wy = -win_half; wy <= win_half; ++wy)
            {
                const int row = (y + wy) * width;
                for (int wx = -win_half; wx <= win_half; ++wx)
                {
                    sum_r_base += pRight[row + (x + wx)];
                }
            }
            float mean_r_base = sum_r_base * inv_win_area;

            float sum_sq_diff_r_base = 0;
            for (int wy = -win_half; wy <= win_half; ++wy)
            {
                const int row = (y + wy) * width;
                for (int wx = -win_half; wx <= win_half; ++wx)
                {
                    float val = static_cast<float>(pRight[row + (x + wx)]) - mean_r_base;
                    sum_sq_diff_r_base += val * val;
                }
            }
            float std_r_base = std::sqrt(sum_sq_diff_r_base);

            // Right left search
            float best_score_r = -1.0f;
            int best_disp_r = 0;
            int max_d_r = std::min(MAX_DISP, width - 1 - (x + win_half));

            for (int d = 0; d <= max_d_r; ++d)
            {
                float sum_l_shift = 0;
                for (int wy = -win_half; wy <= win_half; ++wy)
                {
                    const int row = (y + wy) * width;
                    for (int wx = -win_half; wx <= win_half; ++wx)
                    {
                        sum_l_shift += pLeft[row + (x + d + wx)];
                    }
                }
                float mean_l_shift = sum_l_shift * inv_win_area;

                float sum_cross_r = 0;
                float sum_sq_diff_l_shift = 0;
                for (int wy = -win_half; wy <= win_half; ++wy)
                {
                    const int row = (y + wy) * width;
                    for (int wx = -win_half; wx <= win_half; ++wx)
                    {
                        float val_r_base = static_cast<float>(pRight[row + (x + wx)]) - mean_r_base;
                        float val_l_shift = static_cast<float>(pLeft[row + (x + d + wx)]) - mean_l_shift;
                        sum_cross_r += val_r_base * val_l_shift;
                        sum_sq_diff_l_shift += val_l_shift * val_l_shift;
                    }
                }

                float std_l_shift = std::sqrt(sum_sq_diff_l_shift);
                float den_r = std_r_base * std_l_shift;
                float score_r = (den_r > 0.0001f) ? (sum_cross_r / den_r) : -1.0f;

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

void CrossCheck(const GrayImage &disp_left, const GrayImage &disp_right, GrayImage &output, int width, int height)
{
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int dl = disp_left[y * width + x];
            int x_right = x - dl;
            int dr = (x_right >= 0 && x_right < width) ? disp_right[y * width + x_right] : 0;

            output[y * width + x] = (std::abs(dl - dr) > CROSSCHECK_THRESHOLD) ? 0 : static_cast<std::uint8_t>(dl);
        }
    }
}

void OcclusionFill(const GrayImage &input, GrayImage &output, int width, int height)
{
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            if (input[y * width + x] == 0)
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
            else
            {
                output[y * width + x] = input[y * width + x];
            }
        }
    }
}

void ProcessImageSequential(const RGBAImage &input_rgba, int w, int h, GrayImage &output_gray)
{
    int nw = w / 4, nh = h / 4;
    for (int y = 0; y < nh; ++y)
    {
        for (int x = 0; x < nw; ++x)
        {
            int idx = ((y * 4 * w) + (x * 4)) * 4;
            float r = input_rgba[idx];
            float g = input_rgba[idx + 1];
            float b = input_rgba[idx + 2];
            output_gray[y * nw + x] = static_cast<std::uint8_t>(0.2126f * r + 0.7152f * g + 0.0722f * b);
        }
    }
}

void SaveNormalizedImage(const std::string &filename, const GrayImage &data, int width, int height)
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

int main()
{
    RGBAImage img0_raw, img1_raw;
    unsigned int w, h;

    if (lodepng::decode(img0_raw, w, h, "im0.png") || lodepng::decode(img1_raw, w, h, "im1.png"))
    {
        std::cerr << "Error: Could not load images" << std::endl;
        return 1;
    }

    int nw = static_cast<int>(w) / 4;
    int nh = static_cast<int>(h) / 4;
    size_t new_size = static_cast<size_t>(nw) * nh;

    GrayImage gray0(new_size), gray1(new_size);
    GrayImage d_left(new_size, 0), d_right(new_size, 0);
    GrayImage d_cc(new_size, 0), d_final(new_size, 0);

    std::cout << "Started" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    ProcessImageSequential(img0_raw, w, h, gray0);
    ProcessImageSequential(img1_raw, w, h, gray1);
    SaveNormalizedImage("step1_gray0.png", gray0, nw, nh);
    SaveNormalizedImage("step1_gray1.png", gray1, nw, nh);

    CalcZNCC(gray0, gray1, d_left, d_right, nw, nh);
    SaveNormalizedImage("step2_disp_left.png", d_left, nw, nh);
    SaveNormalizedImage("step2_disp_right.png", d_right, nw, nh);

    CrossCheck(d_left, d_right, d_cc, nw, nh);
    SaveNormalizedImage("step3_crosscheck.png", d_cc, nw, nh);

    OcclusionFill(d_cc, d_final, nw, nh);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Total Time: " << diff.count() << "s" << std::endl;

    SaveNormalizedImage("depthmap.png", d_final, nw, nh);
    std::cout << "Result saved" << std::endl;

    return 0;
}