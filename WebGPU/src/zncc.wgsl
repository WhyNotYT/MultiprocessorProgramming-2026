// WGSL compute shaders for stereo disparity using ZNCC
// passes: grayscale, zncc left/right, cross-check, two-pass occlusion fill

// uniform params shared across all shaders
struct Params {
  width:       u32,   // downscaled width
  height:      u32,   // downscaled height
  srcWidth:    u32,   // original image width
  srcHeight:   u32,   // original image height
  maxDisp:     u32,
  winHalf:     u32,
  crossThresh: u32,
}

@group(0) @binding(0) var<storage, read>       leftRgba:   array<u32>;
@group(0) @binding(1) var<storage, read>       rightRgba:  array<u32>;
@group(0) @binding(2) var<storage, read_write> grayLeft:   array<u32>;
@group(0) @binding(3) var<storage, read_write> grayRight:  array<u32>;
@group(0) @binding(4) var<storage, read_write> dispLeft:   array<u32>;
@group(0) @binding(5) var<storage, read_write> dispRight:  array<u32>;
@group(0) @binding(6) var<storage, read_write> checkedDisp:array<u32>;
@group(0) @binding(7) var<storage, read_write> outputDisp: array<u32>;
@group(0) @binding(8) var<uniform>             params:     Params;
// scratch[y*width*2 + x*2 + 0] = fill value, scratch[...+ 1] = distance (255 = no source)
@group(0) @binding(9) var<storage, read_write> fillScratch: array<u32>;

// convert 2D coords to flat array index
fn idx2d(x: u32, y: u32, w: u32) -> u32 { return y * w + x; }

// unpack RGBA u32 and convert to grayscale using standard luminance weights
fn unpackGray(packed: u32) -> u32 {
  let r = f32( packed        & 0xffu);
  let g = f32((packed >> 8u) & 0xffu);
  let b = f32((packed >> 16u)& 0xffu);
  return u32(0.2126 * r + 0.7152 * g + 0.0722 * b);
}

// converts RGBA to grayscale and downscales 4x in a single pass
// just samples the top-left pixel of each 4x4 block
@compute @workgroup_size(16, 16, 1)
fn grayscale(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let srcX   = gid.x * 4u;
  let srcY   = gid.y * 4u;
  let outIdx = idx2d(gid.x, gid.y, params.width);
  // sample top-left of each 4x4 block for both images
  grayLeft [outIdx] = unpackGray(leftRgba [idx2d(srcX, srcY, params.srcWidth)]);
  grayRight[outIdx] = unpackGray(rightRgba[idx2d(srcX, srcY, params.srcWidth)]);
}

// tile dimensions for shared memory (pixels packed as u8 into u32 words, 4 per word)
// left tile: 16 rows x 40 cols, right tile: 16 rows x 108 cols (wider to cover max disparity)
const TILE_H:         u32 = 16u;
const TILE_W:         u32 = 40u;
const TILE_W_R_COLS:  u32 = 108u;  // padded to multiple of 4
const TILE_L_STRIDE:  u32 = 10u;   // ceil(40 / 4)
const TILE_R_STRIDE:  u32 = 27u;   // ceil(108 / 4)
const TILE_L_SIZE:    u32 = 160u;  // TILE_H * TILE_L_STRIDE
const TILE_R_SIZE:    u32 = 432u;  // TILE_H * TILE_R_STRIDE

// ZNCC left→right disparity
// uses packed u8 shared memory tiles and sliding window row sums to reduce redundant work
var<workgroup> wg_tile_l:   array<atomic<u32>, 160>;  // left tile
var<workgroup> wg_tile_r_L: array<atomic<u32>, 432>;  // right tile (wide)

@compute @workgroup_size(32, 8, 1)
fn znccLeft(
  @builtin(global_invocation_id)   gid:  vec3<u32>,
  @builtin(local_invocation_id)    lid:  vec3<u32>,
  @builtin(workgroup_id)            wgid: vec3<u32>,
) {
  let lx     = lid.x;
  let ly     = lid.y;
  let gx     = gid.x;
  let gy     = gid.y;
  let grp_x  = wgid.x * 32u;
  let grp_y  = wgid.y * 8u;
  let width  = params.width;
  let height = params.height;
  let wh     = params.winHalf;
  let md     = params.maxDisp;
  let win    = 2u * wh + 1u;

  // zero tiles first since atomicOr accumulates
  for (var i = lx + ly * 32u; i < TILE_L_SIZE; i += 256u) {
    atomicStore(&wg_tile_l[i], 0u);
  }
  for (var i = lx + ly * 32u; i < TILE_R_SIZE; i += 256u) {
    atomicStore(&wg_tile_r_L[i], 0u);
  }
  workgroupBarrier();

  // load left tile cooperatively, packing 4 bytes per u32 with atomicOr
  for (var ty = ly; ty < TILE_H; ty += 8u) {
    let sy = clamp(i32(grp_y) - i32(wh) + i32(ty), 0, i32(height) - 1);
    for (var tx = lx; tx < TILE_W; tx += 32u) {
      let sx = clamp(i32(grp_x) - i32(wh) + i32(tx), 0, i32(width) - 1);
      let val = (grayLeft[idx2d(u32(sx), u32(sy), width)] & 0xffu);
      let word  = ty * TILE_L_STRIDE + tx / 4u;
      let shift = (tx % 4u) * 8u;
      atomicOr(&wg_tile_l[word], val << shift);
    }
  }

  // load right tile (extends left by max_disp to cover all search positions)
  for (var ty = ly; ty < TILE_H; ty += 8u) {
    let sy = clamp(i32(grp_y) - i32(wh) + i32(ty), 0, i32(height) - 1);
    for (var tx = lx; tx < TILE_W_R_COLS; tx += 32u) {
      let sx = clamp(i32(grp_x) - i32(wh) - i32(md) + i32(tx), 0, i32(width) - 1);
      let val = (grayRight[idx2d(u32(sx), u32(sy), width)] & 0xffu);
      let word  = ty * TILE_R_STRIDE + tx / 4u;
      let shift = (tx % 4u) * 8u;
      atomicOr(&wg_tile_r_L[word], val << shift);
    }
  }
  workgroupBarrier();

  if (gx >= width || gy >= height) { return; }
  // skip border pixels that don't have a full window
  if (gx < wh || gx >= width  - wh ||
      gy < wh || gy >= height - wh) {
    dispLeft[idx2d(gx, gy, width)] = 0u;
    return;
  }

  let inv_win = 1.0 / f32(win * win);

  // compute left window mean and variance once, reused for all disparities
  var sum_l = 0.0;
  for (var wy = 0u; wy < win; wy++) {
    for (var wx = 0u; wx < win; wx++) {
      let word  = (ly + wy) * TILE_L_STRIDE + (lx + wx) / 4u;
      let shift = ((lx + wx) % 4u) * 8u;
      sum_l += f32((atomicLoad(&wg_tile_l[word]) >> shift) & 0xffu);
    }
  }
  let mean_l = sum_l * inv_win;

  var ssq_l = 0.0;
  for (var wy = 0u; wy < win; wy++) {
    for (var wx = 0u; wx < win; wx++) {
      let word  = (ly + wy) * TILE_L_STRIDE + (lx + wx) / 4u;
      let shift = ((lx + wx) % 4u) * 8u;
      let v = f32((atomicLoad(&wg_tile_l[word]) >> shift) & 0xffu) - mean_l;
      ssq_l += v * v;
    }
  }

  // bootstrap right window row sums at d=0, then slide one column per disparity step
  var row_sum: array<f32, 16>;
  let rb0 = lx + md;
  for (var wy = 0u; wy < win; wy++) {
    var s = 0.0;
    for (var wx = 0u; wx < win; wx++) {
      let col   = rb0 + wx;
      let word  = (ly + wy) * TILE_R_STRIDE + col / 4u;
      let shift = (col % 4u) * 8u;
      s += f32((atomicLoad(&wg_tile_r_L[word]) >> shift) & 0xffu);
    }
    row_sum[wy] = s;
  }

  var best = -1.0;
  var bd   = 0i;
  let lim  = i32(min(md, u32(i32(gx) - i32(wh))));

  for (var d = 0i; d <= lim; d++) {
    let r_base = lx + md - u32(d);

    // slide right window one column left per disparity step
    if (d > 0i) {
      for (var wy = 0u; wy < win; wy++) {
        // drop the rightmost column
        let col_drop  = r_base + win;
        let wd = (ly + wy) * TILE_R_STRIDE + col_drop / 4u;
        let sd = (col_drop % 4u) * 8u;
        row_sum[wy] -= f32((atomicLoad(&wg_tile_r_L[wd]) >> sd) & 0xffu);
        // add the new leftmost column
        let col_add   = r_base;
        let wa = (ly + wy) * TILE_R_STRIDE + col_add / 4u;
        let sa = (col_add % 4u) * 8u;
        row_sum[wy] += f32((atomicLoad(&wg_tile_r_L[wa]) >> sa) & 0xffu);
      }
    }

    var sum_r = 0.0;
    for (var wy = 0u; wy < win; wy++) { sum_r += row_sum[wy]; }
    let mean_r = sum_r * inv_win;

    // compute ZNCC score
    var cross = 0.0;
    var ssq_r = 0.0;
    for (var wy = 0u; wy < win; wy++) {
      for (var wx = 0u; wx < win; wx++) {
        let lword  = (ly + wy) * TILE_L_STRIDE + (lx + wx) / 4u;
        let lshift = ((lx + wx) % 4u) * 8u;
        let vl = f32((atomicLoad(&wg_tile_l[lword]) >> lshift) & 0xffu) - mean_l;

        let rcol   = r_base + wx;
        let rword  = (ly + wy) * TILE_R_STRIDE + rcol / 4u;
        let rshift = (rcol % 4u) * 8u;
        let vr = f32((atomicLoad(&wg_tile_r_L[rword]) >> rshift) & 0xffu) - mean_r;

        cross += vl * vr;
        ssq_r += vr * vr;
      }
    }

    let denom = ssq_l * ssq_r;
    let score = select(-1.0, cross * inverseSqrt(denom + 1e-8), denom > 1e-8);
    if (score > best) { best = score; bd = d; }
  }

  dispLeft[idx2d(gx, gy, width)] = u32(bd);
}

// ZNCC right→left disparity - mirror of znccLeft, left window slides right instead
var<workgroup> wg_tile_r_R:  array<atomic<u32>, 160>;  // narrow right tile
var<workgroup> wg_tile_l_R:  array<atomic<u32>, 432>;  // wide left tile

@compute @workgroup_size(32, 8, 1)
fn znccRight(
  @builtin(global_invocation_id)   gid:  vec3<u32>,
  @builtin(local_invocation_id)    lid:  vec3<u32>,
  @builtin(workgroup_id)            wgid: vec3<u32>,
) {
  let lx     = lid.x;
  let ly     = lid.y;
  let gx     = gid.x;
  let gy     = gid.y;
  let grp_x  = wgid.x * 32u;
  let grp_y  = wgid.y * 8u;
  let width  = params.width;
  let height = params.height;
  let wh     = params.winHalf;
  let md     = params.maxDisp;
  let win    = 2u * wh + 1u;

  // zero tiles before loading
  for (var i = lx + ly * 32u; i < TILE_L_SIZE; i += 256u) {
    atomicStore(&wg_tile_r_R[i], 0u);
  }
  for (var i = lx + ly * 32u; i < TILE_R_SIZE; i += 256u) {
    atomicStore(&wg_tile_l_R[i], 0u);
  }
  workgroupBarrier();

  // load narrow right tile
  for (var ty = ly; ty < TILE_H; ty += 8u) {
    let sy = clamp(i32(grp_y) - i32(wh) + i32(ty), 0, i32(height) - 1);
    for (var tx = lx; tx < TILE_W; tx += 32u) {
      let sx = clamp(i32(grp_x) - i32(wh) + i32(tx), 0, i32(width) - 1);
      let val = (grayRight[idx2d(u32(sx), u32(sy), width)] & 0xffu);
      let word  = ty * TILE_L_STRIDE + tx / 4u;
      let shift = (tx % 4u) * 8u;
      atomicOr(&wg_tile_r_R[word], val << shift);
    }
  }

  // load wide left tile (extends right to cover all disparity offsets)
  for (var ty = ly; ty < TILE_H; ty += 8u) {
    let sy = clamp(i32(grp_y) - i32(wh) + i32(ty), 0, i32(height) - 1);
    for (var tx = lx; tx < TILE_W_R_COLS; tx += 32u) {
      let sx = clamp(i32(grp_x) - i32(wh) + i32(tx), 0, i32(width) - 1);
      let val = (grayLeft[idx2d(u32(sx), u32(sy), width)] & 0xffu);
      let word  = ty * TILE_R_STRIDE + tx / 4u;
      let shift = (tx % 4u) * 8u;
      atomicOr(&wg_tile_l_R[word], val << shift);
    }
  }
  workgroupBarrier();

  if (gx >= width || gy >= height) { return; }
  if (gx < wh || gx >= width  - wh ||
      gy < wh || gy >= height - wh) {
    dispRight[idx2d(gx, gy, width)] = 0u;
    return;
  }

  let inv_win = 1.0 / f32(win * win);

  // right window stats (fixed reference window)
  var sum_r = 0.0;
  for (var wy = 0u; wy < win; wy++) {
    for (var wx = 0u; wx < win; wx++) {
      let word  = (ly + wy) * TILE_L_STRIDE + (lx + wx) / 4u;
      let shift = ((lx + wx) % 4u) * 8u;
      sum_r += f32((atomicLoad(&wg_tile_r_R[word]) >> shift) & 0xffu);
    }
  }
  let mean_r = sum_r * inv_win;

  var ssq_r = 0.0;
  for (var wy = 0u; wy < win; wy++) {
    for (var wx = 0u; wx < win; wx++) {
      let word  = (ly + wy) * TILE_L_STRIDE + (lx + wx) / 4u;
      let shift = ((lx + wx) % 4u) * 8u;
      let v = f32((atomicLoad(&wg_tile_r_R[word]) >> shift) & 0xffu) - mean_r;
      ssq_r += v * v;
    }
  }

  // bootstrap left window row sums at d=0 then slide right
  var row_sum: array<f32, 16>;
  for (var wy = 0u; wy < win; wy++) {
    var s = 0.0;
    for (var wx = 0u; wx < win; wx++) {
      let col   = lx + wx;
      let word  = (ly + wy) * TILE_R_STRIDE + col / 4u;
      let shift = (col % 4u) * 8u;
      s += f32((atomicLoad(&wg_tile_l_R[word]) >> shift) & 0xffu);
    }
    row_sum[wy] = s;
  }

  var best = -1.0;
  var bd   = 0i;
  let lim  = i32(min(md, u32(i32(width) - 1 - (i32(gx) + i32(wh)))));

  for (var d = 0i; d <= lim; d++) {
    let l_base = lx + u32(d);

    // slide left window one column right per disparity step
    if (d > 0i) {
      for (var wy = 0u; wy < win; wy++) {
        // drop leftmost column
        let col_drop  = l_base - 1u;
        let wd = (ly + wy) * TILE_R_STRIDE + col_drop / 4u;
        let sd = (col_drop % 4u) * 8u;
        row_sum[wy] -= f32((atomicLoad(&wg_tile_l_R[wd]) >> sd) & 0xffu);
        // add new rightmost column
        let col_add   = l_base + win - 1u;
        let wa = (ly + wy) * TILE_R_STRIDE + col_add / 4u;
        let sa = (col_add % 4u) * 8u;
        row_sum[wy] += f32((atomicLoad(&wg_tile_l_R[wa]) >> sa) & 0xffu);
      }
    }

    var sum_ls = 0.0;
    for (var wy = 0u; wy < win; wy++) { sum_ls += row_sum[wy]; }
    let mean_ls = sum_ls * inv_win;

    var cross  = 0.0;
    var ssq_ls = 0.0;
    for (var wy = 0u; wy < win; wy++) {
      for (var wx = 0u; wx < win; wx++) {
        let rcol   = lx + wx;
        let rword  = (ly + wy) * TILE_L_STRIDE + rcol / 4u;
        let rshift = (rcol % 4u) * 8u;
        let vr = f32((atomicLoad(&wg_tile_r_R[rword]) >> rshift) & 0xffu) - mean_r;

        let lcol   = l_base + wx;
        let lword  = (ly + wy) * TILE_R_STRIDE + lcol / 4u;
        let lshift = (lcol % 4u) * 8u;
        let vl = f32((atomicLoad(&wg_tile_l_R[lword]) >> lshift) & 0xffu) - mean_ls;

        cross  += vr * vl;
        ssq_ls += vl * vl;
      }
    }

    let denom = ssq_r * ssq_ls;
    let score = select(-1.0, cross * inverseSqrt(denom + 1e-8), denom > 1e-8);
    if (score > best) { best = score; bd = d; }
  }

  dispRight[idx2d(gx, gy, width)] = u32(bd);
}

// zeros out pixels where left and right disparities disagree by more than crossThresh
@compute @workgroup_size(16, 16, 1)
fn crossCheck(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let outIdx = idx2d(gid.x, gid.y, params.width);
  let dl     = i32(dispLeft[outIdx]);
  let xr     = i32(gid.x) - dl;
  var dr     = 0i;
  if (xr >= 0 && xr < i32(params.width)) {
    dr = i32(dispRight[idx2d(u32(xr), gid.y, params.width)]);
  }
  let diff = dl - dr;
  if (diff < -i32(params.crossThresh) || diff > i32(params.crossThresh)) {
    checkedDisp[outIdx] = 0u;
  } else {
    checkedDisp[outIdx] = u32(dl);
  }
}

// pass 1 of occlusion fill: scans left→right per row
// writes the nearest valid disparity and its distance to fillScratch
@compute @workgroup_size(64, 1, 1)
fn fillPass1(@builtin(global_invocation_id) gid: vec3<u32>) {
  let y = gid.x;
  if (y >= params.height) { return; }

  let width    = params.width;
  let row_base = y * width;
  var last_val  = 0u;
  var last_dist = 255u; // 255 = no valid pixel seen yet

  for (var x = 0u; x < width; x++) {
    let v = checkedDisp[row_base + x];
    if (v != 0u) {
      last_val  = v;
      last_dist = 0u;
    } else if (last_dist < 254u) {
      last_dist++;
    }
    let sc = (row_base + x) * 2u;
    fillScratch[sc]      = last_val;
    fillScratch[sc + 1u] = last_dist;
  }
}

// pass 2 of occlusion fill: scans right→left per row
// picks the closer of the left neighbor (from fillScratch) and right neighbor
@compute @workgroup_size(64, 1, 1)
fn fillPass2(@builtin(global_invocation_id) gid: vec3<u32>) {
  let y = gid.x;
  if (y >= params.height) { return; }

  let width    = params.width;
  let row_base = y * width;
  var last_r  = 0u;
  var dist_r  = 255u; // distance to nearest valid pixel on the right

  // WGSL doesn't support decrementing loop variables directly so we do it manually
  var x = width;
  loop {
    if (x == 0u) { break; }
    x--;

    let orig = checkedDisp[row_base + x];
    if (orig != 0u) {
      // pixel already valid, just update right tracking
      outputDisp[row_base + x] = orig;
      last_r = orig;
      dist_r = 0u;
    } else {
      let sc     = (row_base + x) * 2u;
      let lv     = fillScratch[sc];      // nearest-left fill value
      let dist_l = fillScratch[sc + 1u]; // distance to nearest-left

      // pick the closer of left or right neighbor
      var out = 0u;
      if (dist_l == 255u && dist_r == 255u) {
        out = 0u;          // no valid neighbors at all
      } else if (dist_l == 255u) {
        out = last_r;      // only right neighbor exists
      } else if (dist_r == 255u) {
        out = lv;          // only left neighbor exists
      } else {
        out = select(last_r, lv, dist_l <= dist_r); // pick closer one
      }
      outputDisp[row_base + x] = out;

      if (dist_r < 254u) { dist_r++; }
    }
  }
}
