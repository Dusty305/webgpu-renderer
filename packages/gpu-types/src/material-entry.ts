/**
 * Запись материала на стороне GPU, хранящаяся в буфере хранения материалов.
 * Каждый меш ссылается на один MaterialEntry по его идентификатору материала.
 */
export interface MaterialEntry {
  /** Уровень разрешения (0 = 512, 1 = 1024, 2 = 2048) */
  tierIndex: number;
  /** Индекс слоя в texture_2d_array для данного уровня */
  layerIndex: number;
  /** Наиболее детальный уровень мипа, в данный момент находящийся в памяти GPU */
  residentMip: number;
  /**
   * Консервативное значение lodMinClamp для уровня = max(residentMip) по всем текстурам
   * в этом уровне. Шейдер использует это, чтобы выбрать между textureSample (аппаратная
   * анизотропия) и textureSampleLevel (обход LOD для отдельной текстуры).
   */
  tierLodMinClamp: number;
}

/** Определение структуры WGSL, соответствующее MaterialEntry, для включения в шейдеры. */
export const MATERIAL_ENTRY_WGSL = /* wgsl */ `
struct MaterialEntry {
  tierIndex:       u32,
  layerIndex:      u32,
  residentMip:     u32,
  tierLodMinClamp: u32,
}
`;
