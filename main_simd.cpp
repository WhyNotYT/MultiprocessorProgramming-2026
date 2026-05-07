#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#include "lodepng.h"

constexpr int WIN_SIZE = 9;
constexpr int MAX_DISP = 65;
constexpr int CROSSCHECK_THRESHOLD = 8;
constexpr int WIN_HALF = WIN_SIZE / 2;
constexpr int WIN_AREA = WIN_SIZE * WIN_SIZE;

using u8 = std::uint8_t;
using f32 = float;

struct Timer
{
    std::chrono::high_resolution_clock::time_point t0;
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    double ms() const
    {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::high_resolution_clock::now() - t0)
            .count();
    }
};

// ─── horizontal sum of __m256 ─────────────────────────────────────────────────
static inline f32 hsum256(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// ─── resize + grayscale (fused, AVX2) ────────────────────────────────────────
void ResizeGray(const std::vector<u8> &src, int sw, int /*sh*/,
                std::vector<f32> &dst, int dw, int dh)
{
    const __m256 vR = _mm256_set1_ps(0.2126f);
    const __m256 vG = _mm256_set1_ps(0.7152f);
    const __m256 vB = _mm256_set1_ps(0.0722f);

    for (int dy = 0; dy < dh; ++dy)
    {
        const u8 *row = src.data() + dy * 4 * sw * 4; // src stride = sw*4 bytes/row, 4 rows down
        f32 *out = dst.data() + dy * dw;
        int dx = 0;
        for (; dx + 8 <= dw; dx += 8)
        {
            const u8 *p = row + dx * 16; // each src pixel is 4 bytes, sampled every 4 → stride 16
            __m256i ri = _mm256_set_epi32(p[112], p[96], p[80], p[64], p[48], p[32], p[16], p[0]);
            __m256i gi = _mm256_set_epi32(p[113], p[97], p[81], p[65], p[49], p[33], p[17], p[1]);
            __m256i bi = _mm256_set_epi32(p[114], p[98], p[82], p[66], p[50], p[34], p[18], p[2]);
            __m256 gray = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ri), vR,
                                          _mm256_fmadd_ps(_mm256_cvtepi32_ps(gi), vG,
                                                          _mm256_mul_ps(_mm256_cvtepi32_ps(bi), vB)));
            _mm256_storeu_ps(out + dx, gray);
        }
        for (; dx < dw; ++dx)
        {
            const u8 *p = row + dx * 16;
            out[dx] = 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
        }
    }
}

// ─── Integral images (sum and sum-of-squares) for O(1) window queries ─────────
// Stored as double to avoid float overflow on sum-of-squares.
// Layout: (height+1) × (width+1), row-major, top/left border = 0.
struct IntegralImage
{
    std::vector<double> S;  // sum
    std::vector<double> S2; // sum of squares
    int W, H;               // padded dims = (img_w+1, img_h+1)

    void build(const std::vector<f32> &img, int w, int h)
    {
        W = w + 1;
        H = h + 1;
        S.assign(W * H, 0.0);
        S2.assign(W * H, 0.0);
        for (int y = 0; y < h; ++y)
        {
            double rs = 0, rs2 = 0;
            for (int x = 0; x < w; ++x)
            {
                double v = img[y * w + x];
                rs += v;
                rs2 += v * v;
                int idx = (y + 1) * W + (x + 1);
                S[idx] = S[y * W + (x + 1)] + rs;
                S2[idx] = S2[y * W + (x + 1)] + rs2;
            }
        }
    }

    // Rectangle sum: rows [r0,r1], cols [c0,c1]  (inclusive)
    inline double rectSum(const std::vector<double> &T,
                          int r0, int c0, int r1, int c1) const
    {
        return T[(r1 + 1) * W + (c1 + 1)] - T[(r0)*W + (c1 + 1)] - T[(r1 + 1) * W + (c0)] + T[(r0)*W + (c0)];
    }

    inline double winSum(int cy, int cx) const
    {
        return rectSum(S, cy - WIN_HALF, cx - WIN_HALF, cy + WIN_HALF, cx + WIN_HALF);
    }
    inline double winSum2(int cy, int cx) const
    {
        return rectSum(S2, cy - WIN_HALF, cx - WIN_HALF, cy + WIN_HALF, cx + WIN_HALF);
    }
};

// ─── ZNCC using integral images + AVX2 cross-correlation ─────────────────────
//
// Key insight: mean and stddev of BOTH windows are now O(1) via integral images.
// The only remaining O(WIN²) work per (x,d) pair is the cross-correlation sum,
// which we vectorize with AVX2.
//
void CalcZNCC(const std::vector<f32> &left, const std::vector<f32> &right,
              std::vector<u8> &disp_left, std::vector<u8> &disp_right,
              int width, int height)
{
    IntegralImage iiL, iiR;
    iiL.build(left, width, height);
    iiR.build(right, width, height);

    for (int y = WIN_HALF; y < height - WIN_HALF; ++y)
    {
        for (int x = WIN_HALF; x < width - WIN_HALF; ++x)
        {
            // ── left window stats (O(1)) ──────────────────────────────────
            double sumL = iiL.winSum(y, x);
            double sumL2 = iiL.winSum2(y, x);
            f32 meanL = (f32)(sumL / WIN_AREA);
            f32 varL = (f32)(sumL2 / WIN_AREA - (double)meanL * meanL);
            f32 stdL = (varL > 0.f) ? std::sqrt(varL) : 0.f;
            __m256 vML = _mm256_set1_ps(meanL);

            // ── left→right search ─────────────────────────────────────────
            f32 best_l = -2.f;
            int bd_l = 0;
            int max_d_l = std::min(MAX_DISP, x - WIN_HALF);

            for (int d = 0; d <= max_d_l; ++d)
            {
                int rx = x - d;
                double sumR = iiR.winSum(y, rx);
                double sumR2 = iiR.winSum2(y, rx);
                f32 meanR = (f32)(sumR / WIN_AREA);
                f32 varR = (f32)(sumR2 / WIN_AREA - (double)meanR * meanR);
                f32 stdR = (varR > 0.f) ? std::sqrt(varR) : 0.f;
                f32 den = stdL * stdR * WIN_AREA; // denominator

                if (den < 0.0001f)
                    continue;

                // Cross-correlation inner loop (AVX2)
                __m256 vMR = _mm256_set1_ps(meanR);
                __m256 vacc = _mm256_setzero_ps();
                f32 cross = 0.f;

                for (int wy = -WIN_HALF; wy <= WIN_HALF; ++wy)
                {
                    const f32 *rowL = left.data() + (y + wy) * width + (x - WIN_HALF);
                    const f32 *rowR = right.data() + (y + wy) * width + (rx - WIN_HALF);
                    int wx = 0;
                    for (; wx + 8 <= WIN_SIZE; wx += 8)
                    {
                        __m256 dl = _mm256_sub_ps(_mm256_loadu_ps(rowL + wx), vML);
                        __m256 dr = _mm256_sub_ps(_mm256_loadu_ps(rowR + wx), vMR);
                        vacc = _mm256_fmadd_ps(dl, dr, vacc);
                    }
                    for (; wx < WIN_SIZE; ++wx)
                        cross += (rowL[wx] - meanL) * (rowR[wx] - meanR);
                }
                cross += hsum256(vacc);

                f32 score = cross / den;
                if (score > best_l)
                {
                    best_l = score;
                    bd_l = d;
                }
            }
            disp_left[y * width + x] = (u8)bd_l;

            // ── right window stats (O(1)) ─────────────────────────────────
            double sumRb = iiR.winSum(y, x);
            double sumRb2 = iiR.winSum2(y, x);
            f32 meanRb = (f32)(sumRb / WIN_AREA);
            f32 varRb = (f32)(sumRb2 / WIN_AREA - (double)meanRb * meanRb);
            f32 stdRb = (varRb > 0.f) ? std::sqrt(varRb) : 0.f;
            __m256 vMRb = _mm256_set1_ps(meanRb);

            // ── right→left search ─────────────────────────────────────────
            f32 best_r = -2.f;
            int bd_r = 0;
            int max_d_r = std::min(MAX_DISP, width - 1 - (x + WIN_HALF));

            for (int d = 0; d <= max_d_r; ++d)
            {
                int lx = x + d;
                double sumLs = iiL.winSum(y, lx);
                double sumLs2 = iiL.winSum2(y, lx);
                f32 meanLs = (f32)(sumLs / WIN_AREA);
                f32 varLs = (f32)(sumLs2 / WIN_AREA - (double)meanLs * meanLs);
                f32 stdLs = (varLs > 0.f) ? std::sqrt(varLs) : 0.f;
                f32 den_r = stdRb * stdLs * WIN_AREA;

                if (den_r < 0.0001f)
                    continue;

                __m256 vMLs = _mm256_set1_ps(meanLs);
                __m256 vacc = _mm256_setzero_ps();
                f32 cross_r = 0.f;

                for (int wy = -WIN_HALF; wy <= WIN_HALF; ++wy)
                {
                    const f32 *rowR = right.data() + (y + wy) * width + (x - WIN_HALF);
                    const f32 *rowL = left.data() + (y + wy) * width + (lx - WIN_HALF);
                    int wx = 0;
                    for (; wx + 8 <= WIN_SIZE; wx += 8)
                    {
                        __m256 dr = _mm256_sub_ps(_mm256_loadu_ps(rowR + wx), vMRb);
                        __m256 dl = _mm256_sub_ps(_mm256_loadu_ps(rowL + wx), vMLs);
                        vacc = _mm256_fmadd_ps(dr, dl, vacc);
                    }
                    for (; wx < WIN_SIZE; ++wx)
                        cross_r += (rowR[wx] - meanRb) * (rowL[wx] - meanLs);
                }
                cross_r += hsum256(vacc);

                f32 score_r = cross_r / den_r;
                if (score_r > best_r)
                {
                    best_r = score_r;
                    bd_r = d;
                }
            }
            disp_right[y * width + x] = (u8)bd_r;
        }
    }
}

// ─── cross-check ─────────────────────────────────────────────────────────────
void CrossCheck(const std::vector<u8> &dl, const std::vector<u8> &dr,
                std::vector<u8> &out, int width, int height)
{
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            int d = dl[y * width + x];
            int xr = x - d;
            int d2 = (xr >= 0 && xr < width) ? dr[y * width + xr] : 0;
            out[y * width + x] = (std::abs(d - d2) > CROSSCHECK_THRESHOLD) ? 0 : (u8)d;
        }
}

// ─── occlusion fill (two-pass, branch-free inner loop) ───────────────────────
void OcclusionFill(const std::vector<u8> &input, std::vector<u8> &output,
                   int width, int height)
{
    // Pass 1: left→right
    for (int y = 0; y < height; ++y)
    {
        const u8 *in = input.data() + y * width;
        u8 *out = output.data() + y * width;
        u8 last = 0;
        for (int x = 0; x < width; ++x)
        {
            if (in[x])
                last = in[x];
            out[x] = in[x] ? in[x] : last;
        }
    }
    // Pass 2: right→left (only fix remaining zeros)
    for (int y = 0; y < height; ++y)
    {
        u8 *out = output.data() + y * width;
        u8 last = 0;
        for (int x = width - 1; x >= 0; --x)
        {
            if (out[x])
                last = out[x];
            else if (last)
                out[x] = last;
        }
    }
}

// ─── save ─────────────────────────────────────────────────────────────────────
void SaveNormalized(const std::string &fn, const std::vector<u8> &data, int w, int h)
{
    std::vector<u8> rgba(w * h * 4);
    for (int i = 0; i < w * h; ++i)
    {
        u8 v = (u8)((data[i] * 255) / MAX_DISP);
        rgba[i * 4 + 0] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = v;
        rgba[i * 4 + 3] = 255;
    }
    lodepng::encode(fn, rgba, w, h);
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main()
{
    std::vector<u8> img0, img1;
    unsigned int w, h;
    if (lodepng::decode(img0, w, h, "im0.png") || lodepng::decode(img1, w, h, "im1.png"))
    {
        std::cerr << "Error loading images\n";
        return 1;
    }

    int nw = (int)w / 4, nh = (int)h / 4;
    size_t n = (size_t)nw * nh;

    std::vector<f32> gray0(n), gray1(n);
    std::vector<u8> d_left(n, 0), d_right(n, 0), d_cc(n, 0), d_final(n, 0);

    std::cout << "Started\n";
    Timer t;
    double total = 0;
    auto stage = [&](const char *name, auto fn)
    {
        t.start();
        fn();
        double elapsed = t.ms();
        total += elapsed;
        std::cout << "  " << name << ": " << elapsed << " ms\n";
    };

    stage("Resize+Grayscale im0", [&]
          { ResizeGray(img0, w, h, gray0, nw, nh); });
    stage("Resize+Grayscale im1", [&]
          { ResizeGray(img1, w, h, gray1, nw, nh); });
    stage("ZNCC (left+right)   ", [&]
          { CalcZNCC(gray0, gray1, d_left, d_right, nw, nh); });
    stage("Cross-check         ", [&]
          { CrossCheck(d_left, d_right, d_cc, nw, nh); });
    stage("Occlusion fill      ", [&]
          { OcclusionFill(d_cc, d_final, nw, nh); });

    std::cout << "\nTimings summary:\n";
    std::cout << "  Total: " << total << " ms\n";
    SaveNormalized("depthmap.png", d_final, nw, nh);
    std::cout << "Result saved\n";
    return 0;
}