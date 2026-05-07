/* -----------------------------------------------------------------------
 * kernels.cl  –  ZNCC stereo depth, tuned for GTX 1660 Ti (Turing TU116)
 *
 * Optimisation history:
 *  Round 1 (bug fixes):
 *   FIX-1  resize_image / convert_grayscale: bounds guard for padded threads.
 *   FIX-2  zncc_left/right tile load: right/left tile offset by max_disp so
 *          the LDS cache is useful for the full disparity range.
 *   FIX-3  (host) duplicate kernel objects merged.
 *
 *  Round 2 (performance):
 *   OPT-A  occlusion_fill split into two parallel kernels:
 *            fill_pass1: left→right scan, writes (fill_value, distance) pairs
 *                        into a scratch buffer (2 bytes per pixel).
 *            fill_pass2: right→left scan, combines nearest-left and
 *                        nearest-right fill with exact distance comparison.
 *          Replaces O(W²) serial scan with 2 × O(W) parallel passes.
 *          Host change: one extra scratch buffer blf (2 * small_px bytes),
 *          two kernel objects, two enqueue calls instead of one.
 *          Expected speedup: ~300-500x on the fill step.
 *   OPT-B  zncc_left / zncc_right: sliding column-sum keeps right/left
 *          window mean current as d increments in O(win_h) instead of
 *          O(win²).  Expected speedup: ~2-3x on both ZNCC kernels.
 *   OPT-C  ZNCC tiles stored as uchar instead of float → 4x smaller LDS
 *          → ~4x more concurrent workgroups → better latency hiding.
 * ----------------------------------------------------------------------- */

__constant sampler_t sampler =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

/* -----------------------------------------------------------------------
 * resize_image
 * ----------------------------------------------------------------------- */
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

/* -----------------------------------------------------------------------
 * convert_grayscale
 * ----------------------------------------------------------------------- */
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

/* -----------------------------------------------------------------------
 * ZNCC tile geometry  (OPT-C: uchar tiles, 4x smaller LDS)
 *
 *   tile_l : TILE_H x TILE_W        = 16 x  40 =   640 B
 *   tile_r : TILE_H x TILE_W_R      = 16 x 105 = 1680 B
 *   Total per WG                    =           2320 B   (was 9280 B float)
 *   TU116 LDS = 48 KB → up to 20 resident WGs vs ~5 before.
 * ----------------------------------------------------------------------- */
#define WG_X         32
#define WG_Y          8
#define MAX_WIN_HALF  4
#define MAX_DISP_DEF 65
#define TILE_W       (WG_X + 2 * MAX_WIN_HALF)
#define TILE_W_R     (WG_X + 2 * MAX_WIN_HALF + MAX_DISP_DEF)
#define TILE_H       (WG_Y + 2 * MAX_WIN_HALF)

/* -----------------------------------------------------------------------
 * zncc_left
 *
 * Computes disp_left[y][x] = argmax_d ZNCC(left@(x,y), right@(x-d,y))
 *
 * OPT-B: sliding column-sum for right-window mean.
 *   row_sum[wy] = sum of the current right-window column range for row wy.
 *   Bootstrapped at d=0 (O(win^2)), then maintained with one column
 *   add + one column subtract per d increment (O(win_h) per step).
 * ----------------------------------------------------------------------- */
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
    __local uchar tile_l[TILE_H][TILE_W];
    __local uchar tile_r[TILE_H][TILE_W_R];

    const int lx    = get_local_id(0);
    const int ly    = get_local_id(1);
    const int gx    = get_global_id(0);
    const int gy    = get_global_id(1);
    const int grp_x = get_group_id(0) * WG_X;
    const int grp_y = get_group_id(1) * WG_Y;
    const int win   = 2 * win_half + 1;

    for (int ty = ly; ty < TILE_H; ty += WG_Y) {
        int sy = clamp(grp_y - win_half + ty, 0, height - 1);
        for (int tx = lx; tx < TILE_W; tx += WG_X) {
            int sx = clamp(grp_x - win_half + tx, 0, width - 1);
            tile_l[ty][tx] = left[sy * width + sx];
        }
    }
    for (int ty = ly; ty < TILE_H; ty += WG_Y) {
        int sy = clamp(grp_y - win_half + ty, 0, height - 1);
        for (int tx = lx; tx < TILE_W_R; tx += WG_X) {
            int sx = clamp(grp_x - win_half - max_disp + tx, 0, width - 1);
            tile_r[ty][tx] = right[sy * width + sx];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx >= width || gy >= height) return;
    if (gx < win_half || gx >= width  - win_half ||
        gy < win_half || gy >= height - win_half)
    {
        disp_left[gy * width + gx] = 0;
        return;
    }

    const float inv_win = 1.0f / (float)(win * win);

    /* left window stats — computed once */
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

    /* OPT-B: bootstrap right-window row sums at d = 0 */
    float row_sum[TILE_H];
    {
        const int rb0 = lx + max_disp;
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

        /* OPT-B: slide one column */
        if (d > 0) {
            for (int wy = 0; wy < win; wy++) {
                row_sum[wy] -= (float)tile_r[ly + wy][r_base + win];  /* drop right col */
                row_sum[wy] += (float)tile_r[ly + wy][r_base];        /* add  left  col */
            }
        }

        float sum_r = 0.0f;
        for (int wy = 0; wy < win; wy++) sum_r += row_sum[wy];
        const float mean_r = sum_r * inv_win;

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
        float score = (ssq_l * ssq_r > 1e-8f)
                      ? cross * native_rsqrt(ssq_l * ssq_r + 1e-8f)
                      : -1.0f;
        if (score > best) { best = score; bd = d; }
    }

    disp_left[gy * width + gx] = (uchar)bd;
}

/* -----------------------------------------------------------------------
 * zncc_right  – mirror of zncc_left; left window slides right
 * ----------------------------------------------------------------------- */
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
    __local uchar tile_r[TILE_H][TILE_W];
    __local uchar tile_l[TILE_H][TILE_W_R];

    const int lx    = get_local_id(0);
    const int ly    = get_local_id(1);
    const int gx    = get_global_id(0);
    const int gy    = get_global_id(1);
    const int grp_x = get_group_id(0) * WG_X;
    const int grp_y = get_group_id(1) * WG_Y;
    const int win   = 2 * win_half + 1;

    for (int ty = ly; ty < TILE_H; ty += WG_Y) {
        int sy = clamp(grp_y - win_half + ty, 0, height - 1);
        for (int tx = lx; tx < TILE_W; tx += WG_X) {
            int sx = clamp(grp_x - win_half + tx, 0, width - 1);
            tile_r[ty][tx] = right[sy * width + sx];
        }
    }
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

    /* OPT-B: bootstrap left-window row sums at d = 0 */
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

        if (d > 0) {
            for (int wy = 0; wy < win; wy++) {
                row_sum[wy] -= (float)tile_l[ly + wy][l_base - 1];
                row_sum[wy] += (float)tile_l[ly + wy][l_base + win - 1];
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

/* -----------------------------------------------------------------------
 * cross_check  – memory-bound, already fast
 * ----------------------------------------------------------------------- */
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

/* -----------------------------------------------------------------------
 * OPT-A: occlusion_fill, pass 1 of 2  –  fill_pass1
 *
 * One thread per row. Scans left → right.
 * For each pixel, records:
 *   scratch[y * width * 2 + x * 2 + 0]  = fill value  (nearest nonzero to left, or 0)
 *   scratch[y * width * 2 + x * 2 + 1]  = distance    (pixels to that source, clamped 0-254; 255 = none)
 * ----------------------------------------------------------------------- */
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
    uchar last_dist = 255;

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

/* -----------------------------------------------------------------------
 * OPT-A: occlusion_fill, pass 2 of 2  –  fill_pass2
 *
 * One thread per row. Scans right → left.
 * Combines nearest-left fill (from scratch) with nearest-right fill
 * (maintained locally) using exact distance comparison.
 * Nonzero source pixels are written unchanged.
 * ----------------------------------------------------------------------- */
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
    uchar dist_r = 255;

    for (int x = width - 1; x >= 0; x--) {
        uchar orig = row_in[x];
        if (orig != 0) {
            row_out[x] = orig;
            last_r     = orig;
            dist_r     = 0;
        } else {
            uchar lv     = row_sc[x * 2 + 0];
            uchar dist_l = row_sc[x * 2 + 1];

            if      (dist_l == 255 && dist_r == 255) row_out[x] = 0;
            else if (dist_l == 255)                  row_out[x] = last_r;
            else if (dist_r == 255)                  row_out[x] = lv;
            else row_out[x] = (dist_l <= dist_r) ? lv : last_r;

            if (dist_r < 254) dist_r++;
        }
    }
}
