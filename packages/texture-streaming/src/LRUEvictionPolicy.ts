/** Кандидат на вытеснение. */
export interface EvictionCandidate {
  textureId: string;
  mipLevel: number;
  bytes: number;
  lastAccessFrame: number;
}

/**
 * Политика вытеснения LRU для уровней мипмапов GPU-текстур.
 *
 * Правила:
 * - Сначала вытесняются наименее давно использованные уровни мипмапов.
 * - Для одной текстуры сначала вытесняются наиболее детальные мипмапы.
 * - Самые грубые 2 уровня мипмапов никогда не вытесняются (гарантия заглушки).
 */
export class LRUEvictionPolicy {
  /** Отображение "textureId:mipLevel" → индекс последнего обращения по кадрам */
  private readonly _access = new Map<string, number>();
  /** Отображение "textureId:mipLevel" → размер в байтах */
  private readonly _sizes = new Map<string, number>();
  /** Отображение textureId → общее количество мипмапов (для определения самого грубого) */
  private readonly _mipCounts = new Map<string, number>();

  /**
   * Зарегистрировать текстуру, чтобы политика знала количество её мипмапов.
   */
  registerTexture(textureId: string, totalMipLevels: number): void {
    this._mipCounts.set(textureId, totalMipLevels);
  }

  /**
   * Зафиксировать обращение к уровню мипмапа в указанном кадре.
   */
  touch(textureId: string, mipLevel: number, frameIndex: number): void {
    this._access.set(`${textureId}:${mipLevel}`, frameIndex);
  }

  /**
   * Записать размер в байтах для загруженного уровня мипмапа.
   */
  recordSize(textureId: string, mipLevel: number, bytes: number): void {
    this._sizes.set(`${textureId}:${mipLevel}`, bytes);
  }

  /**
   * Забыть уровень мипмапа (после его вытеснения из видеопамяти).
   */
  forget(textureId: string, mipLevel: number): void {
    const k = `${textureId}:${mipLevel}`;
    this._access.delete(k);
    this._sizes.delete(k);
  }

  /**
   * Удалить все данные отслеживания для текстуры (вызывается при полной
   * отмене регистрации текстуры, а не только частичном вытеснении).
   */
  removeTexture(textureId: string): void {
    const totalMips = this._mipCounts.get(textureId) ?? 0;
    for (let level = 0; level < totalMips; level++) {
      const k = `${textureId}:${level}`;
      this._access.delete(k);
      this._sizes.delete(k);
    }
    this._mipCounts.delete(textureId);
  }

  /**
   * Выбрать уровни мипмапов для вытеснения, пока не будет освобождено `bytesNeeded` байт.
   * Возвращает кандидатов в порядке вытеснения (сначала самый детальный мипмап наименее используемой текстуры).
   */
  selectEvictions(bytesNeeded: number): EvictionCandidate[] {
    const candidates: EvictionCandidate[] = [];

    for (const [key, lastAccess] of this._access) {
      const colonIdx = key.lastIndexOf(":");
      const textureId = key.slice(0, colonIdx);
      const mipLevel = parseInt(key.slice(colonIdx + 1), 10);
      const totalMips = this._mipCounts.get(textureId) ?? 1;

      // Никогда не вытеснять самые грубые 2 уровня мипмапов.
      if (mipLevel >= totalMips - 2) continue;

      candidates.push({
        textureId,
        mipLevel,
        bytes: this._sizes.get(key) ?? 0,
        lastAccessFrame: lastAccess,
      });
    }

    // Сортировка: сначала самые старые обращения, затем самые детальные мипмапы при равенстве.
    candidates.sort((a, b) => {
      if (a.lastAccessFrame !== b.lastAccessFrame) {
        return a.lastAccessFrame - b.lastAccessFrame;
      }
      return a.mipLevel - b.mipLevel;
    });

    let freed = 0;
    const selected: EvictionCandidate[] = [];
    for (const candidate of candidates) {
      if (freed >= bytesNeeded) break;
      selected.push(candidate);
      freed += candidate.bytes;
    }
    return selected;
  }

  destroy(): void {
    this._access.clear();
    this._sizes.clear();
    this._mipCounts.clear();
  }
}
