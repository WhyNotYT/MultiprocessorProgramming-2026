// Optimized OpenCL kernels for stereo depth map
// Main improvements over kernels_old.cl:
//   - shared memory tiling reduces global memory reads
//   - sliding column sums for right window mean (OPT-B)
//   - uchar tiles instead of float tiles (4x smaller LDS, more resident WGs) (OPT-C)
//   - two-pass occlusion fill instead of single-pass search (OPT-A)

__constant sampler_t sampler =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

// downscale by 4x with bounds check
__kernel void resize_image(__read_only  image2d_t src,
                           __write_only image2d_t dst)
{
    const int x  = get_global_id(0);
    const int y  = get_global_id(1);
    const int dw = get_image_width(dst);
    const int dh = get_image_height(dst);
    if (x >= dw || y >= dh) return;
    write_imageui(dst, (int2)(x, y),
                  read_imageui(src, sampler, (int2)(x * 4, y * 4)));
}

// convert RGBA image to grayscale with bounds check
__kernel void convert_grayscale(__read_only image2d_t src,
                                __global uchar      *dst,
                                int                  width,
                                int                  height)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height) return;
    uint4 p = read_imageui(src, sampler, (int2)(x, y));
    dst[y * width + x] =
        (uchar)(0.2126f * p.x + 0.7152f * p.y + 0.0722f * p.z);
}

// Tile sizes for shared memory:
//   tile_l : TILE_H x TILE_W        = 16 x  40 =   640 B  (uchar)
//   tile_r : TILE_H x TILE_W_R      = 16 x 105 = 1680 B  (uchar)
//   Total = 2320 B vs 9280 B with floats → ~4x more resident WGs on GPU
#define WG_X         32
#define WG_Y          8
#define MAX_WIN_HALF  4
#define MAX_DISP_DEF 65
#define TILE_W       (WG_X + 2 * MAX_WIN_HALF)        // left tile width
#define TILE_W_R     (WG_X + 2 * MAX_WIN_HALF + MAX_DISP_DEF)  // right tile (wider)
#define TILE_H       (WG_Y + 2 * MAX_WIN_HALF)

// ZNCC left->right disparity
// OPT-B: maintain sliding row sums for the right window so we only do O(win_h) work per d step
// OPT-C: tiles are uchar to fit 4x more data in shared memory
__kernel
__attribute__((reqd_work_group_size(WG_X, WG_Y, 1)))
void zncc_left(__global const uchar *left,
               __global const uchar *right,
               __global uchar       *disp_left,
               int width,
               int height,
               int win_half,
               int max_disp)
{
    __local uchar tile_l[TILE_H][TILE_W];    // left image tile (with halo)
    __local uchar tile_r[TILE_H][TILE_W_R];  // right image tile (wider to cover all disparities)

    const int lx    = get_local_id(0);
    const int ly    = get_local_id(1);
    const int gx    = get_global_id(0);
    const int gy    = get_global_id(1);
    const int grp_x = get_group_id(0) * WG_X;
    const int grp_y = get_group_id(1) * WG_Y;
    const int win   = 2 * win_half + 1;

    // cooperative tile loading - all threads in the WG load the tiles together
    for (int ty = ly; ty < TILE_H; ty += WG_Y) {
        int sy = clamp(grp_y - win_half + ty, 0, height - 1);
        for (int tx = lx; tx < TILE_W; tx += WG_X) {
            int sx = clamp(grp_x - win_half + tx, 0, width - 1);
            tile_l[ty][tx] = left[sy * width + sx];
        }
    }
    // right tile extends left by max_disp to cover all search positions
    for (int ty = ly; ty < TILE_H; ty += WG_Y) {
        int sy = clamp(grp_y - win_half + ty, 0, height - 1);
        for (int tx = lx; tx < TILE_W_R; tx += WG_X) {
            int sx = clamp(grp_x - win_half - max_disp + tx, 0, width - 1);
            tile_r[ty][tx] = right[sy * width + sx];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE); // wait for all threads to finish loading

    if (gx >= width || gy >= height) return;
    if (gx < win_half || gx >= width  - win_half ||
        gy < win_half || gy >= height - win_half)
    {
        disp_left[gy * width + gx] = 0;
        return;
    }

    const float inv_win = 1.0f / (float)(win * win);

    // left window stats - computed once, reused for all disparities
    float sum_l = 0.0f;
    for (int wy = 0; wy < win; wy++)
        for (int wx = 0; wx < win; wx++)
            sum_l += (float)tile_l[ly + wy][lx + wx];
    const float mean_l = sum_l * inv_win;

    float ssq_l = 0.0f;
    for (int wy = 0; wy < win; wy++)
        for (int wx = 0; wx < win; wx++) {
            float v = (float)tile_l[ly + wy][lx + wx] - mean_l;
            ssq_l = fma(v, v, ssq_l);
        }

    // OPT-B: bootstrap right window row sums at d=0 then slide one column at a time
    float row_sum[TILE_H];
    {
        const int rb0 = lx + max_disp; // starting column in tile_r at d=0
        for (int wy = 0; wy < win; wy++) {
            float s = 0.0f;
            for (int wx = 0; wx < win; wx++)
                s += (float)tile_r[ly + wy][rb0 + wx];
            row_sum[wy] = s;
        }
    }

    float best = -1.0f;
    int   bd   = 0;
    const int lim = min(max_disp, gx - win_half);

    for (int d = 0; d <= lim; d++) {
        const int r_base = lx + max_disp - d;

        // OPT-B: update row sums by adding one column and removing another
        if (d > 0) {
            for (int wy = 0; wy < win; wy++) {
                row_sum[wy] -= (float)tile_r[ly + wy][r_base + win];  // drop rightmost col
                row_sum[wy] += (float)tile_r[ly + wy][r_base];        // add new leftmost col
            }
        }

        float sum_r = 0.0f;
        for (int wy = 0; wy < win; wy++) sum_r += row_sum[wy];
        const float mean_r = sum_r * inv_win;

        // cross-correlation and std of right window
        float cross = 0.0f, ssq_r = 0.0f;
        for (int wy = 0; wy < win; wy++) {
            const int row_ly = ly + wy;
            for (int wx = 0; wx < win; wx++) {
                float vl = (float)tile_l[row_ly][lx     + wx] - mean_l;
                float vr = (float)tile_r[row_ly][r_base + wx] - mean_r;
                cross = fma(vl, vr, cross);
                ssq_r = fma(vr, vr, ssq_r);
            }
        }
        // use native_rsqrt for speed (slight precision tradeoff)
        float score = (ssq_l * ssq_r > 1e-8f)
                      ? cross * native_rsqrt(ssq_l * ssq_r + 1e-8f)
                      : -1.0f;
        if (score > best) { best = score; bd = d; }
    }

    disp_left[gy * width + gx] = (uchar)bd;
}

// ZNCC right->left disparity - mirror of zncc_left, left window slides right
__kernel
__attribute__((reqd_work_group_size(WG_X, WG_Y, 1)))
void zncc_right(__global const uchar *left,
                __global const uchar *right,
                __global uchar       *disp_right,
                int width,
                int height,
                int win_half,
                int max_disp)
{
    __local uchar tile_r[TILE_H][TILE_W];    // right image tile (narrow)
    __local uchar tile_l[TILE_H][TILE_W_R];  // left image tile (wider, extends right)

    const int lx    = get_local_id(0);
    const int ly    = get_local_id(1);
    const int gx    = get_global_id(0);
    const int gy    = get_global_id(1);
    const int grp_x = get_group_id(0) * WG_X;
    const int grp_y = get_group_id(1) * WG_Y;
    const int win   = 2 * win_half + 1;

    // load right tile
    for (int ty = ly; ty < TILE_H; ty += WG_Y) {
        int sy = clamp(grp_y - win_half + ty, 0, height - 1);
        for (int tx = lx; tx < TILE_W; tx += WG_X) {
            int sx = clamp(grp_x - win_half + tx, 0, width - 1);
            tile_r[ty][tx] = right[sy * width + sx];
        }
    }
    // load left tile (extends right to cover all d offsets)
    for (int ty = ly; ty < TILE_H; ty += WG_Y) {
        int sy = clamp(grp_y - win_half + ty, 0, height - 1);
        for (int tx = lx; tx < TILE_W_R; tx += WG_X) {
            int sx = clamp(grp_x - win_half + tx, 0, width - 1);
            tile_l[ty][tx] = left[sy * width + sx];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx >= width || gy >= height) return;
    if (gx < win_half || gx >= width  - win_half ||
        gy < win_half || gy >= height - win_half)
    {
        disp_right[gy * width + gx] = 0;
        return;
    }

    const float inv_win = 1.0f / (float)(win * win);

    // right window stats - fixed, computed once
    float sum_r = 0.0f;
    for (int wy = 0; wy < win; wy++)
        for (int wx = 0; wx < win; wx++)
            sum_r += (float)tile_r[ly + wy][lx + wx];
    const float mean_r = sum_r * inv_win;

    float ssq_r = 0.0f;
    for (int wy = 0; wy < win; wy++)
        for (int wx = 0; wx < win; wx++) {
            float v = (float)tile_r[ly + wy][lx + wx] - mean_r;
            ssq_r = fma(v, v, ssq_r);
        }

    // OPT-B: bootstrap left window row sums at d=0
    float row_sum[TILE_H];
    for (int wy = 0; wy < win; wy++) {
        float s = 0.0f;
        for (int wx = 0; wx < win; wx++)
            s += (float)tile_l[ly + wy][lx + wx];
        row_sum[wy] = s;
    }

    float best = -1.0f;
    int   bd   = 0;
    const int lim = min(max_disp, width - 1 - (gx + win_half));

    for (int d = 0; d <= lim; d++) {
        const int l_base = lx + d;

        // OPT-B: slide left window one column to the right
        if (d > 0) {
            for (int wy = 0; wy < win; wy++) {
                row_sum[wy] -= (float)tile_l[ly + wy][l_base - 1];           // drop leftmost col
                row_sum[wy] += (float)tile_l[ly + wy][l_base + win - 1];     // add new rightmost col
            }
        }

        float sum_ls = 0.0f;
        for (int wy = 0; wy < win; wy++) sum_ls += row_sum[wy];
        const float mean_ls = sum_ls * inv_win;

        float cross = 0.0f, ssq_ls = 0.0f;
        for (int wy = 0; wy < win; wy++) {
            const int row_ly = ly + wy;
            for (int wx = 0; wx < win; wx++) {
                float vr = (float)tile_r[row_ly][lx     + wx] - mean_r;
                float vl = (float)tile_l[row_ly][l_base + wx] - mean_ls;
                cross   = fma(vr, vl, cross);
                ssq_ls  = fma(vl, vl, ssq_ls);
            }
        }
        float score = (ssq_r * ssq_ls > 1e-8f)
                      ? cross * native_rsqrt(ssq_r * ssq_ls + 1e-8f)
                      : -1.0f;
        if (score > best) { best = score; bd = d; }
    }

    disp_right[gy * width + gx] = (uchar)bd;
}

// zero out pixels where left/right disparities disagree (memory-bound, already fast)
__kernel void cross_check(__global const uchar *disp_left,
                          __global const uchar *disp_right,
                          __global uchar       *output,
                          int width,
                          int height,
                          int threshold)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int dl      = (int)disp_left[y * width + x];
    int x_right = x - dl;
    int dr      = (x_right >= 0 && x_right < width)
                      ? (int)disp_right[y * width + x_right] : 0;
    int diff    = dl - dr;
    output[y * width + x] =
        (diff < -threshold || diff > threshold) ? 0 : (uchar)dl;
}

// OPT-A: occlusion fill pass 1 - scan left to right
// records nearest-left fill value and distance for each pixel in scratch buffer
// scratch layout: [fill_val, distance_to_source] per pixel, 2 bytes each
__kernel void fill_pass1(__global const uchar *input,
                         __global uchar       *scratch,
                         int width,
                         int height)
{
    int y = get_global_id(0);
    if (y >= height) return;

    const __global uchar *row_in = input   + y * width;
    __global uchar       *row_sc = scratch + y * width * 2;

    uchar last_val  = 0;
    uchar last_dist = 255; // 255 = no valid pixel seen yet

    for (int x = 0; x < width; x++) {
        if (row_in[x] != 0) {
            last_val  = row_in[x];
            last_dist = 0;
        } else if (last_dist < 254) {
            last_dist++;
        }
        row_sc[x * 2 + 0] = last_val;
        row_sc[x * 2 + 1] = last_dist;
    }
}

// OPT-A: occlusion fill pass 2 - scan right to left
// combines nearest-left (from scratch) and nearest-right (tracked locally)
// picks the closer one; nonzero pixels are written unchanged
__kernel void fill_pass2(__global const uchar *input,
                         __global const uchar *scratch,
                         __global uchar       *output,
                         int width,
                         int height)
{
    int y = get_global_id(0);
    if (y >= height) return;

    const __global uchar *row_in = input   + y * width;
    const __global uchar *row_sc = scratch + y * width * 2;
    __global uchar       *row_out = output + y * width;

    uchar last_r = 0;
    uchar dist_r = 255; // distance to nearest valid pixel on the right

    for (int x = width - 1; x >= 0; x--) {
        uchar orig = row_in[x];
        if (orig != 0) {
            // pixel is valid, write as-is and update right tracking
            row_out[x] = orig;
            last_r     = orig;
            dist_r     = 0;
        } else {
            uchar lv     = row_sc[x * 2 + 0]; // nearest-left value
            uchar dist_l = row_sc[x * 2 + 1]; // distance to nearest-left

            // pick nearest neighbor, fallback to 0 if none exists
            if      (dist_l == 255 && dist_r == 255) row_out[x] = 0;
            else if (dist_l == 255)                  row_out[x] = last_r;
            else if (dist_r == 255)                  row_out[x] = lv;
            else row_out[x] = (dist_l <= dist_r) ? lv : last_r;

            if (dist_r < 254) dist_r++;
        }
    }
}
