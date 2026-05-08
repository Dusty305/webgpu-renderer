/**
 * Дескриптор ресурса, зарегистрированного в ResourceRegistry.
 */
export interface ResourceDescriptor {
  /** Удобочитаемая метка для отладки */
  label?: string;
}

/**
 * Центральный реестр для совместного использования GPU-ресурсов (групп привязок, буферов и т.д.)
 * между IResourceManager и IRenderPass без жёстких зависимостей.
 */
export interface ResourceRegistry {
  /**
   * Зарегистрировать именованный ресурс, чтобы другие плагины могли его получить.
   */
  register<T>(name: string, resource: T, descriptor?: ResourceDescriptor): void;

  /**
   * Получить зарегистрированный ресурс по имени. Возвращает null, если не найден.
   */
  request<T>(name: string): T | null;

  /**
   * Удалить ранее зарегистрированный ресурс.
   */
  deregister(name: string): void;

  /**
   * Подписаться на изменения для конкретного имени ресурса.
   * Колбэк вызывается при каждой регистрации или отмене регистрации ресурса.
   */
  onChange(name: string, callback: (resource: unknown) => void): () => void;
}
