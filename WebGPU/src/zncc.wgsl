// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------
struct Params {
  width:       u32,   // downscaled width
  height:      u32,   // downscaled height
  srcWidth:    u32,   // original width
  srcHeight:   u32,   // original height
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
// Layout: scratch[y*width*2 + x*2 + 0] = fill value (u8 stored as u32)
//         scratch[y*width*2 + x*2 + 1] = distance   (u8 stored as u32, 255=none)
@group(0) @binding(9) var<storage, read_write> fillScratch: array<u32>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
fn idx2d(x: u32, y: u32, w: u32) -> u32 { return y * w + x; }

fn unpackGray(packed: u32) -> u32 {
  let r = f32( packed        & 0xffu);
  let g = f32((packed >> 8u) & 0xffu);
  let b = f32((packed >> 16u)& 0xffu);
  return u32(0.2126 * r + 0.7152 * g + 0.0722 * b);
}

// ---------------------------------------------------------------------------
// grayscale  –  16×16 workgroup (unchanged logic, tuned WG size)
// Combines resize (4× downscale) + RGB→grey in one pass.
// ---------------------------------------------------------------------------
@compute @workgroup_size(16, 16, 1)
fn grayscale(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let srcX   = gid.x * 4u;
  let srcY   = gid.y * 4u;
  let outIdx = idx2d(gid.x, gid.y, params.width);
  grayLeft [outIdx] = unpackGray(leftRgba [idx2d(srcX, srcY, params.srcWidth)]);
  grayRight[outIdx] = unpackGray(rightRgba[idx2d(srcX, srcY, params.srcWidth)]);
}

// ---------------------------------------------------------------------------
// Packed storage: each row is stored in ceil(TILE_W / 4) u32s.
//   tile_l row stride  = ceil(40  / 4) = 10 u32s  → total 16*10 = 160 u32s
//   tile_r row stride  = ceil(108 / 4) = 27 u32s  → total 16*27 = 432 u32s
//
// To read pixel (ty, tx) from tile_l:  tile_l[ty*10 + tx/4] >> ((tx%4)*8) & 0xff
// ---------------------------------------------------------------------------
const TILE_H:         u32 = 16u;   // WG_Y + 2*MAX_WIN_HALF
const TILE_W:         u32 = 40u;   // WG_X + 2*MAX_WIN_HALF
const TILE_W_R_COLS:  u32 = 108u;  // WG_X + 2*MAX_WIN_HALF + MAX_DISP, padded to mult of 4
const TILE_L_STRIDE:  u32 = 10u;   // ceil(40  / 4)
const TILE_R_STRIDE:  u32 = 27u;   // ceil(108 / 4)
const TILE_L_SIZE:    u32 = 160u;  // TILE_H * TILE_L_STRIDE
const TILE_R_SIZE:    u32 = 432u;  // TILE_H * TILE_R_STRIDE

// Note: tile access is done inline throughout the kernels using atomicOr /
// atomicLoad directly, so no helper functions are needed here.

// ---------------------------------------------------------------------------
// znccLeft  –  32×8 workgroup
//
// ---------------------------------------------------------------------------
var<workgroup> wg_tile_l:  array<atomic<u32>, 160>;  // TILE_L_SIZE = TILE_H * TILE_L_STRIDE
var<workgroup> wg_tile_r_L: array<atomic<u32>, 432>;  // TILE_R_SIZE = TILE_H * TILE_R_STRIDE

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

  // --- Zero shared tiles (each thread clears a few words) ---
  for (var i = lx + ly * 32u; i < TILE_L_SIZE; i += 256u) {
    atomicStore(&wg_tile_l[i], 0u);
  }
  for (var i = lx + ly * 32u; i < TILE_R_SIZE; i += 256u) {
    atomicStore(&wg_tile_r_L[i], 0u);
  }
  workgroupBarrier();

  // --- Load left tile (OPT-C: packed u8) ---
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

  // --- Load right tile (wider, offset left by max_disp; OPT-C: packed u8) ---
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
  if (gx < wh || gx >= width  - wh ||
      gy < wh || gy >= height - wh) {
    dispLeft[idx2d(gx, gy, width)] = 0u;
    return;
  }

  let inv_win = 1.0 / f32(win * win);

  // --- Left window stats (computed once) ---
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

  // --- OPT-B: bootstrap right-window row sums at d=0 ---
  // row_sum[wy] = sum of columns [rb0 .. rb0+win) in tile_r row (ly+wy)
  var row_sum: array<f32, 16>;  // TILE_H = 16, indexed [0..win)
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

    // OPT-B: slide one column left
    if (d > 0i) {
      for (var wy = 0u; wy < win; wy++) {
        // drop rightmost column of previous window
        let col_drop  = r_base + win;
        let wd = (ly + wy) * TILE_R_STRIDE + col_drop / 4u;
        let sd = (col_drop % 4u) * 8u;
        row_sum[wy] -= f32((atomicLoad(&wg_tile_r_L[wd]) >> sd) & 0xffu);
        // add new leftmost column
        let col_add   = r_base;
        let wa = (ly + wy) * TILE_R_STRIDE + col_add / 4u;
        let sa = (col_add % 4u) * 8u;
        row_sum[wy] += f32((atomicLoad(&wg_tile_r_L[wa]) >> sa) & 0xffu);
      }
    }

    var sum_r = 0.0;
    for (var wy = 0u; wy < win; wy++) { sum_r += row_sum[wy]; }
    let mean_r = sum_r * inv_win;

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

// ---------------------------------------------------------------------------
// znccRight  –  32×8 workgroup, mirror of znccLeft; left window slides right
//
// ---------------------------------------------------------------------------
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

  // --- Zero shared tiles ---
  for (var i = lx + ly * 32u; i < TILE_L_SIZE; i += 256u) {
    atomicStore(&wg_tile_r_R[i], 0u);
  }
  for (var i = lx + ly * 32u; i < TILE_R_SIZE; i += 256u) {
    atomicStore(&wg_tile_l_R[i], 0u);
  }
  workgroupBarrier();

  // --- Load narrow right tile ---
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

  // --- Load wide left tile (extends right by max_disp) ---
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

  // --- Right window stats (computed once) ---
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

  // --- OPT-B: bootstrap left-window row sums at d=0 ---
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

    // OPT-B: slide one column right
    if (d > 0i) {
      for (var wy = 0u; wy < win; wy++) {
        // drop leftmost column of previous window
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

// ---------------------------------------------------------------------------
// crossCheck  –  16×16 workgroup (memory-bound, already fast)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Scans left→right. Writes to fillScratch:
//   fillScratch[y*width*2 + x*2 + 0] = nearest nonzero to left  (fill value)
//   fillScratch[y*width*2 + x*2 + 1] = distance to that source  (255 = none)
// ---------------------------------------------------------------------------
@compute @workgroup_size(64, 1, 1)
fn fillPass1(@builtin(global_invocation_id) gid: vec3<u32>) {
  let y = gid.x;
  if (y >= params.height) { return; }

  let width    = params.width;
  let row_base = y * width;
  var last_val  = 0u;
  var last_dist = 255u;

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

// ---------------------------------------------------------------------------
//
// Scans right→left. Combines nearest-left (from fillScratch) with
// nearest-right (maintained locally) using exact distance comparison.
// Nonzero source pixels are written unchanged.
// ---------------------------------------------------------------------------
@compute @workgroup_size(64, 1, 1)
fn fillPass2(@builtin(global_invocation_id) gid: vec3<u32>) {
  let y = gid.x;
  if (y >= params.height) { return; }

  let width    = params.width;
  let row_base = y * width;
  var last_r  = 0u;
  var dist_r  = 255u;

  var x = width;
  loop {
    if (x == 0u) { break; }
    x--;

    let orig = checkedDisp[row_base + x];
    if (orig != 0u) {
      outputDisp[row_base + x] = orig;
      last_r = orig;
      dist_r = 0u;
    } else {
      let sc     = (row_base + x) * 2u;
      let lv     = fillScratch[sc];
      let dist_l = fillScratch[sc + 1u];

      var out = 0u;
      if (dist_l == 255u && dist_r == 255u) {
        out = 0u;
      } else if (dist_l == 255u) {
        out = last_r;
      } else if (dist_r == 255u) {
        out = lv;
      } else {
        out = select(last_r, lv, dist_l <= dist_r);
      }
      outputDisp[row_base + x] = out;

      if (dist_r < 254u) { dist_r++; }
    }
  }
}
