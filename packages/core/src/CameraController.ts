import type { CameraState } from "@webgpu-streaming/gpu-types";

/** Минимальная единичная матрица mat4. */
function mat4Identity(): Float32Array {
  // prettier-ignore
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);
}

/** Перемножить две матрицы 4x4, хранящиеся по столбцам. */
function mat4Multiply(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        sum += a[k * 4 + j]! * b[i * 4 + k]!;
      }
      out[i * 4 + j] = sum;
    }
  }
  return out;
}

/** Матрица перспективной проекции (хранится по столбцам). */
function mat4Perspective(fovY: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1.0 / Math.tan(fovY / 2);
  const rangeInv = 1 / (near - far);
  // prettier-ignore
  return new Float32Array([
    f / aspect, 0, 0,                        0,
    0,          f, 0,                        0,
    0,          0, (far + near) * rangeInv,  -1,
    0,          0, 2 * far * near * rangeInv, 0,
  ]);
}

/**
 * Простой контроллер орбитальной камеры.
 * Прикрепляет обработчики мыши/касания к холсту для интерактивного вращения.
 */
export class CameraController {
  private _theta = 0;     // азимут (радианы)
  private _phi = 0.4;     // угол возвышения (радианы)
  private _radius = 3;
  private _target = new Float32Array([0, 0, 0]);
  private _fovY = Math.PI / 4;
  private _near = 0.1;
  private _far = 1000;
  private _aspect = 1;

  private _pointerDown = false;
  private _lastX = 0;
  private _lastY = 0;

  private _onPointerDown: (e: PointerEvent) => void;
  private _onPointerMove: (e: PointerEvent) => void;
  private _onPointerUp: (e: PointerEvent) => void;
  private _onWheel: (e: WheelEvent) => void;

  constructor(private readonly _canvas: HTMLCanvasElement) {
    this._onPointerDown = (e) => {
      this._pointerDown = true;
      this._lastX = e.clientX;
      this._lastY = e.clientY;
      _canvas.setPointerCapture(e.pointerId);
    };
    this._onPointerMove = (e) => {
      if (!this._pointerDown) return;
      const dx = e.clientX - this._lastX;
      const dy = e.clientY - this._lastY;
      this._lastX = e.clientX;
      this._lastY = e.clientY;
      this._theta += dx * 0.01;
      this._phi = Math.max(-Math.PI / 2 + 0.05, Math.min(Math.PI / 2 - 0.05, this._phi - dy * 0.01));
    };
    this._onPointerUp = () => {
      this._pointerDown = false;
    };
    this._onWheel = (e) => {
      this._radius = Math.max(0.5, this._radius * (1 + e.deltaY * 0.001));
    };

    _canvas.addEventListener("pointerdown", this._onPointerDown);
    _canvas.addEventListener("pointermove", this._onPointerMove);
    _canvas.addEventListener("pointerup", this._onPointerUp);
    _canvas.addEventListener("wheel", this._onWheel, { passive: true });
  }

  setAspect(width: number, height: number): void {
    this._aspect = width / height;
  }

  /** Установить радиус орбиты (расстояние от цели). Минимум 0.5. */
  setRadius(r: number): void {
    this._radius = Math.max(0.5, r);
  }

  /** Установить точку цели орбиты. */
  setTarget(x: number, y: number, z: number): void {
    this._target[0] = x; this._target[1] = y; this._target[2] = z;
  }

  /** Установить поле зрения (радианы). */
  setFov(radians: number): void { this._fovY = radians; }

  /** Установить ближнюю/дальнюю плоскости отсечения. */
  setClip(near: number, far: number): void { this._near = near; this._far = far; }

  /** Вычислить и вернуть текущее состояние камеры. */
  getCameraState(): CameraState {
    const x = this._target[0]! + this._radius * Math.cos(this._phi) * Math.sin(this._theta);
    const y = this._target[1]! + this._radius * Math.sin(this._phi);
    const z = this._target[2]! + this._radius * Math.cos(this._phi) * Math.cos(this._theta);

    const position = new Float32Array([x, y, z]);

    // Определять соотношение сторон из элемента холста каждый кадр, чтобы изменение
    // размера отражалось автоматически без явного вызова setAspect().
    const aspect = (this._canvas.width > 0 && this._canvas.height > 0)
      ? this._canvas.width / this._canvas.height
      : this._aspect;

    // Вычислить матрицу вида (look-at)
    const viewMatrix = this._lookAt(position, this._target);
    const projectionMatrix = mat4Perspective(this._fovY, aspect, this._near, this._far);
    const viewProjectionMatrix = mat4Multiply(projectionMatrix, viewMatrix);

    return {
      viewMatrix,
      projectionMatrix,
      viewProjectionMatrix,
      position,
      fovY: this._fovY,
      near: this._near,
      far: this._far,
      viewportWidth: 0,   // переопределяется в WebGPUElement реальным размером холста
      viewportHeight: 0,
    };
  }

  private _lookAt(eye: Float32Array, center: Float32Array): Float32Array {
    const up = new Float32Array([0, 1, 0]);
    const f = this._normalize(new Float32Array([
      center[0]! - eye[0]!,
      center[1]! - eye[1]!,
      center[2]! - eye[2]!,
    ]));
    const r = this._normalize(this._cross(f, up));
    const u = this._cross(r, f);

    const m = mat4Identity();
    m[0] = r[0]!;  m[4] = r[1]!;  m[8]  = r[2]!;
    m[1] = u[0]!;  m[5] = u[1]!;  m[9]  = u[2]!;
    m[2] = -f[0]!; m[6] = -f[1]!; m[10] = -f[2]!;
    m[12] = -(r[0]! * eye[0]! + r[1]! * eye[1]! + r[2]! * eye[2]!);
    m[13] = -(u[0]! * eye[0]! + u[1]! * eye[1]! + u[2]! * eye[2]!);
    m[14] =  (f[0]! * eye[0]! + f[1]! * eye[1]! + f[2]! * eye[2]!);
    return m;
  }

  private _normalize(v: Float32Array): Float32Array {
    const len = Math.sqrt(v[0]! * v[0]! + v[1]! * v[1]! + v[2]! * v[2]!);
    return new Float32Array([v[0]! / len, v[1]! / len, v[2]! / len]);
  }

  private _cross(a: Float32Array, b: Float32Array): Float32Array {
    return new Float32Array([
      a[1]! * b[2]! - a[2]! * b[1]!,
      a[2]! * b[0]! - a[0]! * b[2]!,
      a[0]! * b[1]! - a[1]! * b[0]!,
    ]);
  }

  destroy(): void {
    this._canvas.removeEventListener("pointerdown", this._onPointerDown);
    this._canvas.removeEventListener("pointermove", this._onPointerMove);
    this._canvas.removeEventListener("pointerup", this._onPointerUp);
    this._canvas.removeEventListener("wheel", this._onWheel);
  }
}
