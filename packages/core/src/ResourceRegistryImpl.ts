import type { ResourceRegistry, ResourceDescriptor } from "@webgpu-streaming/gpu-types";

/**
 * Конкретная реализация ResourceRegistry.
 * Позволяет менеджерам ресурсов и рендер-проходам совместно использовать GPU-ресурсы
 * по имени без прямых зависимостей между импортами.
 */
export class ResourceRegistryImpl implements ResourceRegistry {
  private readonly _resources = new Map<string, unknown>();
  private readonly _listeners = new Map<string, Set<(resource: unknown) => void>>();

  register<T>(name: string, resource: T, _descriptor?: ResourceDescriptor): void {
    this._resources.set(name, resource);
    this._notify(name, resource);
  }

  request<T>(name: string): T | null {
    const value = this._resources.get(name);
    return value !== undefined ? (value as T) : null;
  }

  deregister(name: string): void {
    this._resources.delete(name);
    this._notify(name, undefined);
  }

  onChange(name: string, callback: (resource: unknown) => void): () => void {
    if (!this._listeners.has(name)) {
      this._listeners.set(name, new Set());
    }
    this._listeners.get(name)!.add(callback);

    return () => {
      this._listeners.get(name)?.delete(callback);
    };
  }

  private _notify(name: string, resource: unknown): void {
    for (const cb of this._listeners.get(name) ?? []) {
      cb(resource);
    }
  }
}
