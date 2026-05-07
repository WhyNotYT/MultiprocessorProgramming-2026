import "./style.css";
import { disparityToImageData, runZncc } from "./zncc";

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) {
  throw new Error("Missing #app element.");
}

app.innerHTML = `
  <main class="container">
    <h1>ZNCC Stereo (WebGPU)</h1>
    <p class="subtitle">Upload left/right images, run ZNCC on GPU, and preview disparity.</p>

    <section class="inputs">
      <label>
        Left image
        <input id="leftInput" type="file" accept="image/*" />
      </label>
      <label>
        Right image
        <input id="rightInput" type="file" accept="image/*" />
      </label>
      <button id="runBtn" type="button">Run ZNCC</button>
    </section>

    <p id="status" class="status">Waiting for input.</p>

    <section class="preview">
      <figure>
        <figcaption>Left</figcaption>
        <img id="leftPreview" alt="Left preview" />
      </figure>
      <figure>
        <figcaption>Right</figcaption>
        <img id="rightPreview" alt="Right preview" />
      </figure>
      <figure>
        <figcaption>Disparity</figcaption>
        <canvas id="resultCanvas"></canvas>
      </figure>
    </section>
  </main>
`;

const leftInput = document.querySelector<HTMLInputElement>("#leftInput");
const rightInput = document.querySelector<HTMLInputElement>("#rightInput");
const runBtn = document.querySelector<HTMLButtonElement>("#runBtn");
const leftPreview = document.querySelector<HTMLImageElement>("#leftPreview");
const rightPreview = document.querySelector<HTMLImageElement>("#rightPreview");
const resultCanvas = document.querySelector<HTMLCanvasElement>("#resultCanvas");
const statusEl = document.querySelector<HTMLParagraphElement>("#status");

if (!leftInput || !rightInput || !runBtn || !leftPreview || !rightPreview || !resultCanvas || !statusEl) {
  throw new Error("UI initialization failed.");
}

let leftBitmap: ImageBitmap | null = null;
let rightBitmap: ImageBitmap | null = null;

function setStatus(message: string): void {
  statusEl!.textContent = message;
}

async function fileToBitmap(file: File): Promise<ImageBitmap> {
  return createImageBitmap(file);
}

async function onInputChanged(input: HTMLInputElement, side: "left" | "right"): Promise<void> {
  const file = input.files?.[0];
  if (!file) {
    return;
  }
  const bitmap = await fileToBitmap(file);
  const objectUrl = URL.createObjectURL(file);

  if (side === "left") {
    leftBitmap = bitmap;
      leftPreview!.src = objectUrl;
  } else {
    rightBitmap = bitmap;
      rightPreview!.src = objectUrl;
  }
  setStatus("Images loaded. Ready to run.");
}

function bitmapToImageData(bitmap: ImageBitmap): ImageData {
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Failed to create 2D context.");
  }
  ctx.drawImage(bitmap, 0, 0);
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

leftInput.addEventListener("change", async () => onInputChanged(leftInput, "left"));
rightInput.addEventListener("change", async () => onInputChanged(rightInput, "right"));

runBtn.addEventListener("click", async () => {
  try {
    if (!leftBitmap || !rightBitmap) {
      throw new Error("Please upload both images first.");
    }
    if (leftBitmap.width !== rightBitmap.width || leftBitmap.height !== rightBitmap.height) {
      throw new Error("Left and right images must have identical dimensions.");
    }

    setStatus("Running ZNCC on WebGPU...");
    runBtn.disabled = true;

    const leftData = bitmapToImageData(leftBitmap);
    const rightData = bitmapToImageData(rightBitmap);
    const result = await runZncc(leftData, rightData);

    const usedFallback = result.nonZeroCount === 0 && result.nonZeroRawLeftCount > 0;
    const displayDisp = usedFallback ? result.disparityLeftRaw : result.disparity;
    const output = disparityToImageData(displayDisp, result.width, result.height);
    resultCanvas.width = output.width;
    resultCanvas.height = output.height;
    const ctx = resultCanvas.getContext("2d");
    if (!ctx) {
      throw new Error("Failed to create canvas context.");
    }
    ctx.putImageData(output, 0, 0);

    const mode = usedFallback ? "raw left disparity fallback" : "post-processed disparity";
    setStatus(
      `Done (${mode}). Non-zero: ${result.nonZeroCount}/${result.disparity.length}, raw-left non-zero: ${result.nonZeroRawLeftCount}/${result.disparityLeftRaw.length}, GPU: ${result.elapsedMs.toFixed(2)} ms`
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    setStatus(`Error: ${message} Check browser WebGPU/Vulkan settings.`);
  } finally {
    runBtn.disabled = false;
  }
});
