/**
 * CPU-side KTX2 mip pyramid builder.
 * Converts raw RGBA8 pixel data into a KTX2-compatible ArrayBuffer
 * suitable for TextureStreamingManager.registerTexture().
 */

// ---- Mip generation -------------------------------------------------------

function downsampleMip(src: Uint8Array, sw: number, sh: number): Uint8Array {
  const dw = Math.max(1, sw >> 1);
  const dh = Math.max(1, sh >> 1);
  const dst = new Uint8Array(dw * dh * 4);
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      const so  = (y * 2 * sw + x * 2) * 4;
      const do_ = (y * dw + x) * 4;
      for (let c = 0; c < 4; c++) {
        const sx1 = x * 2 + 1 < sw ? so + 4 + c : so + c;
        const sy1 = y * 2 + 1 < sh ? so + sw * 4 + c : so + c;
        const sxy = x * 2 + 1 < sw && y * 2 + 1 < sh ? so + sw * 4 + 4 + c : so + c;
        dst[do_ + c] = ((src[so + c]! + src[sx1]! + src[sy1]! + src[sxy]!) >> 2);
      }
    }
  }
  return dst;
}

export interface MipPyramidResult {
  mips: Uint8Array[];
  widths: number[];
  heights: number[];
}

export function buildMipPyramid(
  mip0: Uint8Array,
  width: number,
  height: number,
): MipPyramidResult {
  const mips: Uint8Array[] = [mip0];
  const widths = [width];
  const heights = [height];
  let cw = width, ch = height, cur = mip0;
  while (cw > 1 || ch > 1) {
    const next = downsampleMip(cur, cw, ch);
    cw = Math.max(1, cw >> 1);
    ch = Math.max(1, ch >> 1);
    mips.push(next);
    widths.push(cw);
    heights.push(ch);
    cur = next;
  }
  return { mips, widths, heights };
}

// ---- KTX2 packing ---------------------------------------------------------

const KTX2_MAGIC = new Uint8Array([
  0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a,
]);

/** Packs a mip pyramid into an uncompressed RGBA8 SRGB KTX2 buffer. */
export function packKtx2(
  mips: Uint8Array[],
  width: number,
  height: number,
): ArrayBuffer {
  const mc = mips.length;
  const DFD_OFFSET = 80 + mc * 24;
  const DFD_SIZE = 28;
  const MIPS_START = DFD_OFFSET + DFD_SIZE;

  const offsets: number[] = [];
  let off = MIPS_START;
  for (let l = 0; l < mc; l++) {
    offsets.push(off);
    off += mips[l]!.byteLength;
  }

  const buf = new ArrayBuffer(off);
  const u8  = new Uint8Array(buf);
  const dv  = new DataView(buf);

  u8.set(KTX2_MAGIC, 0);
  let p = 12;
  dv.setUint32(p, 43, true); p += 4; // VK_FORMAT_R8G8B8A8_SRGB
  dv.setUint32(p,  1, true); p += 4; // typeSize
  dv.setUint32(p, width,  true); p += 4;
  dv.setUint32(p, height, true); p += 4;
  dv.setUint32(p, 0, true); p += 4;  // pixelDepth
  dv.setUint32(p, 0, true); p += 4;  // layerCount
  dv.setUint32(p, 1, true); p += 4;  // faceCount
  dv.setUint32(p, mc, true); p += 4;
  dv.setUint32(p, 0, true); p += 4;  // supercompressionScheme
  dv.setUint32(p, DFD_OFFSET, true); p += 4;
  dv.setUint32(p, DFD_SIZE, true); p += 4;
  dv.setUint32(p, 0, true); p += 4;  // kvd offset
  dv.setUint32(p, 0, true); p += 4;  // kvd length
  dv.setBigUint64(p, 0n, true); p += 8; // sgd offset
  dv.setBigUint64(p, 0n, true); p += 8; // sgd length

  for (let l = 0; l < mc; l++) {
    const bl = BigInt(mips[l]!.byteLength);
    dv.setBigUint64(p, BigInt(offsets[l]!), true); p += 8;
    dv.setBigUint64(p, bl, true); p += 8;
    dv.setBigUint64(p, bl, true); p += 8;
  }

  // Minimal DFD (Data Format Descriptor)
  dv.setUint32(p, DFD_SIZE, true); p += 4;
  dv.setUint32(p, DFD_SIZE - 4, true); p += 4;
  for (let i = 0; i < 5; i++) { dv.setUint32(p, 0, true); p += 4; }

  for (let l = 0; l < mc; l++) u8.set(mips[l]!, offsets[l]!);
  return buf;
}

// ---- Image decoding -------------------------------------------------------

/**
 * Decodes an image Blob to raw RGBA8 pixels via OffscreenCanvas.
 * Scales the image to fit the nearest power-of-two tier size (512/1024/2048)
 * so it fills its texture array layer exactly.
 */
export async function decodeImage(
  blob: Blob,
): Promise<{ data: Uint8Array; width: number; height: number }> {
  const bitmap = await createImageBitmap(blob);
  const { width, height } = bitmap;
  const maxDim = Math.max(width, height);
  const tierSize = maxDim <= 512 ? 512 : maxDim <= 1024 ? 1024 : 2048;

  const canvas = new OffscreenCanvas(tierSize, tierSize);
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(bitmap, 0, 0, tierSize, tierSize);
  bitmap.close();

  const imageData = ctx.getImageData(0, 0, tierSize, tierSize);
  return { data: new Uint8Array(imageData.data.buffer), width: tierSize, height: tierSize };
}
