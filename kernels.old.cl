// OpenCL kernels for stereo depth map pipeline
// Includes: resize, grayscale, gaussian filter, ZNCC, cross-check, occlusion fill

// sampler for reading images: no normalization, clamp edges, nearest-neighbor
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
                               CLK_ADDRESS_CLAMP_TO_EDGE | 
                               CLK_FILTER_NEAREST;

// downscale image by 4x - each output pixel reads from (x*4, y*4) in the source
__kernel void resize_image(__read_only  image2d_t input_image,
                           __write_only image2d_t output_image)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    int2 input_coord = (int2)(x * 4, y * 4);
    uint4 pixel = read_imageui(input_image, sampler, input_coord);
    write_imageui(output_image, (int2)(x, y), pixel);
}

// convert RGBA image to grayscale using standard luminance weights
__kernel void convert_grayscale(__read_only image2d_t input_image,
                                __global uchar *output_buffer,
                                int width)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    uint4 pixel = read_imageui(input_image, sampler, (int2)(x, y));
    float gray  = 0.2126f * pixel.x + 0.7152f * pixel.y + 0.0722f * pixel.z;
    output_buffer[y * width + x] = (uchar)gray;
}

// 5x5 gaussian blur (sum/256 normalization, not actually used in the main pipeline)
__kernel void apply_filter(__global const uchar *input,
                           __global uchar *output,
                           int width,
                           int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    // gaussian kernel weights (sum = 256)
    const int kernel_weights[25] = {
        1,  4,  6,  4, 1,
        4, 16, 24, 16, 4,
        6, 24, 36, 24, 6,
        4, 16, 24, 16, 4,
        1,  4,  6,  4, 1
    };

    if (x >= 2 && x < width - 2 && y >= 2 && y < height - 2) {
        int sum   = 0;
        int k_idx = 0;
        for (int j = -2; j <= 2; j++) {
            for (int i = -2; i <= 2; i++) {
                sum += input[(y + j) * width + (x + i)] * kernel_weights[k_idx];
                k_idx++;
            }
        }
        output[y * width + x] = (uchar)(sum >> 8); // divide by 256
    } else {
        output[y * width + x] = input[y * width + x]; // border: copy as-is
    }
}

// ZNCC stereo matching - computes both left->right and right->left disparities
// each thread handles one pixel and searches through all disparities
__kernel void zncc(__global const uchar *left,
                   __global const uchar *right,
                   __global uchar       *disp_left,
                   __global uchar       *disp_right,
                   int width,
                   int height,
                   int win_half,
                   int max_disp)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);

    // skip border pixels where the window would go out of bounds
    if (x < win_half || x >= width  - win_half ||
        y < win_half || y >= height - win_half)
        return;

    const float inv_win = 1.0f / (float)((2*win_half+1) * (2*win_half+1));

    // compute left window mean
    float sum_l = 0.0f;
    for (int wy = -win_half; wy <= win_half; wy++) {
        int row = (y + wy) * width;
        for (int wx = -win_half; wx <= win_half; wx++)
            sum_l += left[row + (x + wx)];
    }
    float mean_l = sum_l * inv_win;

    // compute left window std
    float ssq_l = 0.0f;
    for (int wy = -win_half; wy <= win_half; wy++) {
        int row = (y + wy) * width;
        for (int wx = -win_half; wx <= win_half; wx++) {
            float v = (float)left[row + (x + wx)] - mean_l;
            ssq_l += v * v;
        }
    }
    float std_l = sqrt(ssq_l);

    // precompute right window stats at d=0 (used for right->left search)
    float sum_r_base = 0.0f;
    for (int wy = -win_half; wy <= win_half; wy++) {
        int row = (y + wy) * width;
        for (int wx = -win_half; wx <= win_half; wx++)
            sum_r_base += right[row + (x + wx)];
    }
    float mean_r_base = sum_r_base * inv_win;

    float ssq_r_base = 0.0f;
    for (int wy = -win_half; wy <= win_half; wy++) {
        int row = (y + wy) * width;
        for (int wx = -win_half; wx <= win_half; wx++) {
            float v = (float)right[row + (x + wx)] - mean_r_base;
            ssq_r_base += v * v;
        }
    }
    float std_r_base = sqrt(ssq_r_base);

    // left->right: find best matching disparity
    float best_l = -1.0f;
    int   bd_l   = 0;
    int   lim_l  = min(max_disp, x - win_half);

    for (int d = 0; d <= lim_l; d++) {
        float sum_r = 0.0f;
        for (int wy = -win_half; wy <= win_half; wy++) {
            int row = (y + wy) * width;
            for (int wx = -win_half; wx <= win_half; wx++)
                sum_r += right[row + (x - d + wx)];
        }
        float mean_r = sum_r * inv_win;

        float cross = 0.0f, ssq_r = 0.0f;
        for (int wy = -win_half; wy <= win_half; wy++) {
            int row = (y + wy) * width;
            for (int wx = -win_half; wx <= win_half; wx++) {
                float vl = (float)left[row  + (x + wx)]     - mean_l;
                float vr = (float)right[row + (x - d + wx)] - mean_r;
                cross  += vl * vr;
                ssq_r  += vr * vr;
            }
        }
        float std_r = sqrt(ssq_r);
        float den   = std_l * std_r;
        float score = (den > 1e-4f) ? (cross / den) : -1.0f;
        if (score > best_l) { best_l = score; bd_l = d; }
    }
    disp_left[y * width + x] = (uchar)bd_l;

    // right->left: find best matching disparity
    float best_r = -1.0f;
    int   bd_r   = 0;
    int   lim_r  = min(max_disp, width - 1 - (x + win_half));

    for (int d = 0; d <= lim_r; d++) {
        float sum_ls = 0.0f;
        for (int wy = -win_half; wy <= win_half; wy++) {
            int row = (y + wy) * width;
            for (int wx = -win_half; wx <= win_half; wx++)
                sum_ls += left[row + (x + d + wx)];
        }
        float mean_ls = sum_ls * inv_win;

        float cross = 0.0f, ssq_ls = 0.0f;
        for (int wy = -win_half; wy <= win_half; wy++) {
            int row = (y + wy) * width;
            for (int wx = -win_half; wx <= win_half; wx++) {
                float vr = (float)right[row + (x + wx)]      - mean_r_base;
                float vl = (float)left[row  + (x + d + wx)]  - mean_ls;
                cross   += vr * vl;
                ssq_ls  += vl * vl;
            }
        }
        float std_ls = sqrt(ssq_ls);
        float den_r  = std_r_base * std_ls;
        float score  = (den_r > 1e-4f) ? (cross / den_r) : -1.0f;
        if (score > best_r) { best_r = score; bd_r = d; }
    }
    disp_right[y * width + x] = (uchar)bd_r;
}

// zero out pixels where left and right disparities don't agree
__kernel void cross_check(__global const uchar *disp_left,
                          __global const uchar *disp_right,
                          __global uchar       *output,
                          int width,
                          int height,
                          int threshold)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= width || y >= height) return;

    int dl      = disp_left[y * width + x];
    int x_right = x - dl;
    int dr      = (x_right >= 0 && x_right < width)
                      ? (int)disp_right[y * width + x_right]
                      : 0;
    int diff    = dl - dr;
    output[y * width + x] =
        (diff < -threshold || diff > threshold) ? 0 : (uchar)dl;
}

// fill invalid (zero) pixels with nearest non-zero neighbor on the same row
// one thread per row, sequential within each row
__kernel void occlusion_fill(__global const uchar *input,
                             __global uchar       *output,
                             int width,
                             int height)
{
    int y = (int)get_global_id(0);
    if (y >= height) return;

    for (int x = 0; x < width; x++) {
        if (input[y * width + x] != 0) {
            output[y * width + x] = input[y * width + x];
        } else {
            // search left and right alternately for nearest valid pixel
            uchar fill = 0;
            for (int off = 1; off < width; off++) {
                if (x - off >= 0 && input[y * width + (x - off)] != 0) {
                    fill = input[y * width + (x - off)]; break;
                }
                if (x + off < width && input[y * width + (x + off)] != 0) {
                    fill = input[y * width + (x + off)]; break;
                }
            }
            output[y * width + x] = fill;
        }
    }
}
