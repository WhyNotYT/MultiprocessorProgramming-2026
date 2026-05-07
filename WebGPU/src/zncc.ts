/// <reference types="@webgpu/types" />
import shaderCode from "./zncc.wgsl?raw";

// ---------------------------------------------------------------------------
// Constants  (must match the WGSL)
// ---------------------------------------------------------------------------
const WIN_SIZE = 9; // must be odd; win_half = WIN_SIZE/2 = 4
const MAX_DISP = 65;
const CROSSCHECK_THRESH = 8;
const DOWNSCALE = 4;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export type ZnccResult = {
  width: number;
  height: number;
  disparity: Uint8Array; // final (cross-checked + filled)
  disparityLeftRaw: Uint8Array; // raw left disparity before cross-check
  nonZeroCount: number;
  nonZeroRawLeftCount: number;
  elapsedMs: number;
};

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function assertWebGpu(): GPU {
  if (!("gpu" in navigator)) {
    throw new Error(
      "WebGPU API is unavailable in this browser. " +
        "Try Chromium 121+ or Firefox Nightly with WebGPU enabled.",
    );
  }
  if (!window.isSecureContext) {
    throw new Error("WebGPU requires a secure context (https or localhost).");
  }
  return navigator.gpu;
}

function createStorageBuffer(
  device: GPUDevice,
  byteLength: number,
  mapped = false,
): GPUBuffer {
  return device.createBuffer({
    size: byteLength,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
    mappedAtCreation: mapped,
  });
}

// ---------------------------------------------------------------------------
// Stage labels (must stay in sync with dispatch order below)
// ---------------------------------------------------------------------------
const STAGE_LABELS = [
  "CPU → GPU transfer",
  "grayscale",
  "zncc left",
  "zncc right",
  "cross_check",
  "occlusion_fill pass 1",
  "occlusion_fill pass 2",
  "GPU → CPU transfer",
] as const;

// ---------------------------------------------------------------------------
// runZncc
// ---------------------------------------------------------------------------
export async function runZncc(
  leftImage: ImageData,
  rightImage: ImageData,
): Promise<ZnccResult> {
  if (
    leftImage.width !== rightImage.width ||
    leftImage.height !== rightImage.height
  ) {
    throw new Error("Input images must have the same dimensions.");
  }
  if (leftImage.width % DOWNSCALE !== 0 || leftImage.height % DOWNSCALE !== 0) {
    throw new Error("Image width and height must be divisible by 4.");
  }

  const gpu = assertWebGpu();
  const adapter = await gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) {
    throw new Error(
      "No WebGPU adapter found. On Linux, ensure hardware acceleration " +
        "and Vulkan are enabled in your browser.",
    );
  }


  // Log adapter info — adapter.info is the current spec (Chrome 121+).
  // adapter.requestAdapterInfo() was removed in newer builds.
  const adapterInfo: GPUAdapterInfo =
    (adapter as any).info ?? (await (adapter as any).requestAdapterInfo?.()) ?? {} as GPUAdapterInfo;
  console.log(
    `[WebGPU] adapter: ${adapterInfo.description || adapterInfo.device || "(unknown)"}` +
    ` | vendor: ${adapterInfo.vendor || "—"}` +
    ` | arch: ${adapterInfo.architecture || "—"}`,
  );

  // We use 9 storage buffers per stage (WebGPU default limit is 8).
  // We also request timestamp-query for per-pass GPU timing.
  const supportsTimestamps = adapter.features.has("timestamp-query");
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBuffersPerShaderStage: 9,
    },
    requiredFeatures: supportsTimestamps
      ? (["timestamp-query"] as GPUFeatureName[])
      : [],
  });

  const srcWidth = leftImage.width;
  const srcHeight = leftImage.height;
  const width = Math.floor(srcWidth / DOWNSCALE);
  const height = Math.floor(srcHeight / DOWNSCALE);
  const pixelCount = width * height;

  // ---------------------------------------------------------------------------
  // Timestamp query setup
  // ---------------------------------------------------------------------------
  // We place a begin/end timestamp around each compute pass.
  // Passes: grayscale, znccLeft, znccRight, crossCheck, fillPass1, fillPass2
  // That's 6 passes × 2 timestamps = 12 slots.
  const NUM_PASSES = 6;
  const NUM_TIMESTAMPS = NUM_PASSES * 2; // begin + end per pass

  let querySet: GPUQuerySet | null = null;
  let tsResolveBuffer: GPUBuffer | null = null;
  let tsReadbackBuffer: GPUBuffer | null = null;

  if (supportsTimestamps) {
    querySet = device.createQuerySet({
      type: "timestamp",
      count: NUM_TIMESTAMPS,
    });
    tsResolveBuffer = device.createBuffer({
      size: NUM_TIMESTAMPS * 8, // each timestamp is a u64 (8 bytes)
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    tsReadbackBuffer = device.createBuffer({
      size: NUM_TIMESTAMPS * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  // ---------------------------------------------------------------------------
  // Upload source RGBA images — measure CPU→GPU transfer time
  // ---------------------------------------------------------------------------
  const tCpuGpuStart = performance.now();

  const leftRgba = new Uint8Array(leftImage.data);
  const rightRgba = new Uint8Array(rightImage.data);

  const leftBuffer = createStorageBuffer(device, leftRgba.byteLength, true);
  new Uint8Array(leftBuffer.getMappedRange()).set(leftRgba);
  leftBuffer.unmap();

  const rightBuffer = createStorageBuffer(device, rightRgba.byteLength, true);
  new Uint8Array(rightBuffer.getMappedRange()).set(rightRgba);
  rightBuffer.unmap();

  // Flush the upload queue and wait so the transfer time is accurate.
  device.queue.submit([]);
  await device.queue.onSubmittedWorkDone();

  const tCpuGpuEnd = performance.now();
  const cpuGpuMs = tCpuGpuEnd - tCpuGpuStart;

  // --- Intermediate / output buffers ---
  const grayLeft = createStorageBuffer(device, pixelCount * 4);
  const grayRight = createStorageBuffer(device, pixelCount * 4);
  const dispLeft = createStorageBuffer(device, pixelCount * 4);
  const dispRight = createStorageBuffer(device, pixelCount * 4);
  const checkedDisp = createStorageBuffer(device, pixelCount * 4);
  const outputDisp = createStorageBuffer(device, pixelCount * 4);

  // OPT-A: scratch buffer for fill pass 1 → pass 2 handoff.
  // 2 u32s per pixel: [fill_value, distance_to_source].
  const fillScratch = createStorageBuffer(device, pixelCount * 2 * 4);

  // --- Uniform params ---
  const paramsArray = new Uint32Array([
    width,
    height,
    srcWidth,
    srcHeight,
    MAX_DISP,
    Math.floor(WIN_SIZE / 2), // winHalf = 4
    CROSSCHECK_THRESH,
  ]);
  const paramsBuffer = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, paramsArray);

  // --- Shader module ---
  const shaderModule = device.createShaderModule({ code: shaderCode });

  // Check for compilation errors and surface them clearly.
  if (shaderModule.getCompilationInfo) {
    const info = await shaderModule.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === "error");
    if (errors.length > 0) {
      const msg = errors.map((e) => `  line ${e.lineNum}: ${e.message}`).join("\n");
      throw new Error(`WGSL compilation failed:\n${msg}`);
    }
  }

  // --- Bind group layout (10 bindings: 0-8 same as before + binding 9 = fillScratch) ---
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 6,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 7,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 8,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
      // OPT-A: fill scratch buffer (binding 9 — requires maxStorageBuffersPerShaderStage: 9)
      {
        binding: 9,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  // --- Pipelines ---
  // grayscale: 16×16
  const grayscalePipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "grayscale" },
  });

  // OPT-B + OPT-C: separate znccLeft and znccRight at 32×8
  const znccLeftPipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "znccLeft" },
  });
  const znccRightPipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "znccRight" },
  });

  // crossCheck: 16×16
  const crossCheckPipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "crossCheck" },
  });

  // OPT-A: two fill passes at 64×1
  const fillPass1Pipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "fillPass1" },
  });
  const fillPass2Pipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "fillPass2" },
  });

  // --- Bind group ---
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: leftBuffer } },
      { binding: 1, resource: { buffer: rightBuffer } },
      { binding: 2, resource: { buffer: grayLeft } },
      { binding: 3, resource: { buffer: grayRight } },
      { binding: 4, resource: { buffer: dispLeft } },
      { binding: 5, resource: { buffer: dispRight } },
      { binding: 6, resource: { buffer: checkedDisp } },
      { binding: 7, resource: { buffer: outputDisp } },
      { binding: 8, resource: { buffer: paramsBuffer } },
      { binding: 9, resource: { buffer: fillScratch } }, // OPT-A
    ],
  });

  // ---------------------------------------------------------------------------
  // Helper: wrap a compute pass with optional timestamp begin/end
  // passIndex: 0-based index into the 6 compute passes
  // ---------------------------------------------------------------------------
  function timedPass(
    encoder: GPUCommandEncoder,
    passIndex: number,
    fn: (pass: GPUComputePassEncoder) => void,
  ) {
    const tsSlot = passIndex * 2;
    const passDescriptor: GPUComputePassDescriptor =
      querySet && supportsTimestamps
        ? {
            timestampWrites: {
              querySet,
              beginningOfPassWriteIndex: tsSlot,
              endOfPassWriteIndex: tsSlot + 1,
            },
          }
        : {};
    const pass = encoder.beginComputePass(passDescriptor);
    fn(pass);
    pass.end();
  }

  // --- Dispatch ---
  const tKernelStart = performance.now();
  const encoder = device.createCommandEncoder();

  // 0. Grayscale + downscale  (16×16 WG)
  timedPass(encoder, 0, (pass) => {
    pass.setPipeline(grayscalePipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
  });

  // 1. ZNCC left  (32×8 WG, OPT-B + OPT-C)
  timedPass(encoder, 1, (pass) => {
    pass.setPipeline(znccLeftPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(width / 32), Math.ceil(height / 8));
  });

  // 2. ZNCC right  (32×8 WG, OPT-B + OPT-C)
  timedPass(encoder, 2, (pass) => {
    pass.setPipeline(znccRightPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(width / 32), Math.ceil(height / 8));
  });

  // 3. Cross-check  (16×16 WG)
  timedPass(encoder, 3, (pass) => {
    pass.setPipeline(crossCheckPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
  });

  // 4. OPT-A: fill pass 1 – left→right  (64×1 WG, one thread per row)
  timedPass(encoder, 4, (pass) => {
    pass.setPipeline(fillPass1Pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(height / 64));
  });

  // 5. OPT-A: fill pass 2 – right→left  (64×1 WG, one thread per row)
  timedPass(encoder, 5, (pass) => {
    pass.setPipeline(fillPass2Pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(height / 64));
  });

  // Resolve timestamps into the resolve buffer before readback copies.
  if (querySet && tsResolveBuffer) {
    encoder.resolveQuerySet(querySet, 0, NUM_TIMESTAMPS, tsResolveBuffer, 0);
    if (tsReadbackBuffer) {
      encoder.copyBufferToBuffer(
        tsResolveBuffer,
        0,
        tsReadbackBuffer,
        0,
        NUM_TIMESTAMPS * 8,
      );
    }
  }

  // --- Readback ---
  const readbackOutput = device.createBuffer({
    size: pixelCount * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const readbackLeft = device.createBuffer({
    size: pixelCount * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  encoder.copyBufferToBuffer(outputDisp, 0, readbackOutput, 0, pixelCount * 4);
  encoder.copyBufferToBuffer(dispLeft, 0, readbackLeft, 0, pixelCount * 4);

  const tGpuCpuStart = performance.now();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const tGpuCpuEnd = performance.now();

  const kernelTotalMs = tGpuCpuEnd - tKernelStart;
  const gpuCpuMs = tGpuCpuEnd - tGpuCpuStart;

  await Promise.all([
    readbackOutput.mapAsync(GPUMapMode.READ),
    readbackLeft.mapAsync(GPUMapMode.READ),
    tsReadbackBuffer ? tsReadbackBuffer.mapAsync(GPUMapMode.READ) : Promise.resolve(),
  ]);

  // ---------------------------------------------------------------------------
  // Parse GPU timestamps
  // ---------------------------------------------------------------------------
  const passGpuMs: number[] = new Array(NUM_PASSES).fill(0);
  if (supportsTimestamps && tsReadbackBuffer) {
    const tsData = new BigInt64Array(tsReadbackBuffer.getMappedRange());
    for (let i = 0; i < NUM_PASSES; i++) {
      const begin = tsData[i * 2];
      const end   = tsData[i * 2 + 1];
      // timestamps are in nanoseconds
      passGpuMs[i] = Number(end - begin) / 1_000_000;
    }
    tsReadbackBuffer.unmap();
  }

  const outputRaw = new Uint32Array(readbackOutput.getMappedRange());
  const leftRaw = new Uint32Array(readbackLeft.getMappedRange());
  const disparity = new Uint8Array(pixelCount);
  const disparityLeftRaw = new Uint8Array(pixelCount);
  let nonZeroCount = 0;
  let nonZeroRawLeftCount = 0;

  for (let i = 0; i < pixelCount; i++) {
    disparity[i] = outputRaw[i] & 0xff;
    disparityLeftRaw[i] = leftRaw[i] & 0xff;
    if (disparity[i] !== 0) nonZeroCount++;
    if (disparityLeftRaw[i] !== 0) nonZeroRawLeftCount++;
  }

  readbackOutput.unmap();
  readbackLeft.unmap();

  const totalMs = cpuGpuMs + kernelTotalMs;

  // ---------------------------------------------------------------------------
  // Print timing table
  // ---------------------------------------------------------------------------
  const gpuKernelTotalMs = passGpuMs.reduce((a, b) => a + b, 0);

  // Pass indices: 0=grayscale, 1=znccLeft, 2=znccRight, 3=crossCheck, 4=fill1, 5=fill2
  const rows: [string, string][] = [
    ["CPU → GPU transfer",  `${cpuGpuMs.toFixed(6)} ms`],
    ["grayscale",           supportsTimestamps ? `${passGpuMs[0].toFixed(6)} ms` : "n/a"],
    ["zncc left",           supportsTimestamps ? `${passGpuMs[1].toFixed(6)} ms` : "n/a"],
    ["zncc right",          supportsTimestamps ? `${passGpuMs[2].toFixed(6)} ms` : "n/a"],
    ["cross_check",         supportsTimestamps ? `${passGpuMs[3].toFixed(6)} ms` : "n/a"],
    ["occlusion_fill pass 1", supportsTimestamps ? `${passGpuMs[4].toFixed(6)} ms` : "n/a"],
    ["occlusion_fill pass 2", supportsTimestamps ? `${passGpuMs[5].toFixed(6)} ms` : "n/a"],
    ["GPU → CPU transfer",  `${gpuCpuMs.toFixed(6)} ms`],
    ["**GPU kernels total**", supportsTimestamps ? `**${gpuKernelTotalMs.toFixed(4)} ms**` : `**${kernelTotalMs.toFixed(4)} ms** (wall)`],
    ["**Total**",           `**${totalMs.toFixed(4)} ms**`],
  ];

  const colW = Math.max(...rows.map(([label]) => label.length));
  const valW = Math.max(...rows.map(([, val]) => val.length));

  const sep = `| ${"-".repeat(colW)} | ${"-".repeat(valW)} |`;
  const header = `| ${"Stage".padEnd(colW)} | ${"Time".padEnd(valW)} |`;

  console.log("\nZNCC timing");
  console.log(header);
  console.log(sep);
  for (const [label, val] of rows) {
    console.log(`| ${label.padEnd(colW)} | ${val.padEnd(valW)} |`);
  }
  if (!supportsTimestamps) {
    console.log(
      "\nNote: GPU timestamp-query not supported on this device. " +
        "Per-kernel times unavailable; only wall-clock totals are shown.",
    );
  }

  return {
    width,
    height,
    disparity,
    disparityLeftRaw,
    nonZeroCount,
    nonZeroRawLeftCount,
    elapsedMs: totalMs,
  };
}

// ---------------------------------------------------------------------------
// disparityToImageData  (unchanged)
// ---------------------------------------------------------------------------
export function disparityToImageData(
  disparity: Uint8Array,
  width: number,
  height: number,
): ImageData {
  const rgba = new Uint8ClampedArray(width * height * 4);
  let min = 255,
    max = 0;
  for (const v of disparity) {
    if (v !== 0) {
      min = Math.min(min, v);
      max = Math.max(max, v);
    }
  }
  const hasRange = max > min;
  for (let i = 0; i < disparity.length; i++) {
    const value = disparity[i];
    let normalized = 0;
    if (hasRange && value !== 0) {
      normalized = Math.round(((value - min) * 255) / (max - min));
    } else {
      normalized = Math.round((value * 255) / MAX_DISP);
    }
    const base = i * 4;
    rgba[base] = rgba[base + 1] = rgba[base + 2] = normalized;
    rgba[base + 3] = 255;
  }
  return new ImageData(rgba, width, height);
}
