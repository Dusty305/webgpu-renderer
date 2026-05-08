import type { SceneGraphReadView, SceneNode } from "@webgpu-streaming/gpu-types";

/** Изменяемая версия SceneNode для внутреннего использования. */
interface MutableSceneNode {
  id: string;
  worldTransform: Float32Array;
  boundingSphere: Float32Array;
  materialId: number;
  visible: boolean;
}

/**
 * Минимальный граф сцены для отслеживания объектов меша, преобразований и идентификаторов материалов.
 * Предоставляет представление только для чтения плагинам через SceneGraphReadView.
 */
export class SceneGraph {
  private readonly _nodes = new Map<string, MutableSceneNode>();

  /**
   * Добавить объект меша в граф сцены.
   * @param id - Уникальный идентификатор
   * @param materialId - Индекс в буфере материалов
   * @param worldTransform - Мировое преобразование 4x4, хранящееся по столбцам (Float32Array из 16 элементов)
   * @param boundingSphere - [cx, cy, cz, radius] (Float32Array из 4 элементов)
   */
  addNode(
    id: string,
    materialId: number,
    worldTransform: Float32Array,
    boundingSphere: Float32Array
  ): void {
    this._nodes.set(id, {
      id,
      worldTransform: new Float32Array(worldTransform),
      boundingSphere: new Float32Array(boundingSphere),
      materialId,
      visible: true,
    });
  }

  /** Обновить мировое преобразование существующего узла. */
  updateTransform(id: string, worldTransform: Float32Array): void {
    const node = this._nodes.get(id);
    if (node) {
      node.worldTransform.set(worldTransform);
    }
  }

  /** Установить видимость узла. */
  setVisible(id: string, visible: boolean): void {
    const node = this._nodes.get(id);
    if (node) {
      node.visible = visible;
    }
  }

  /** Удалить узел из графа сцены. */
  removeNode(id: string): void {
    this._nodes.delete(id);
  }

  /** Возвращает снимок представления только для чтения для плагинов. */
  getReadView(): SceneGraphReadView {
    const nodes: SceneNode[] = Array.from(this._nodes.values()).map((n) => ({
      id: n.id,
      worldTransform: n.worldTransform,
      boundingSphere: n.boundingSphere,
      materialId: n.materialId,
      visible: n.visible,
    }));
    return { nodes };
  }

  /** Общее число узлов в сцене. */
  get size(): number {
    return this._nodes.size;
  }
}
