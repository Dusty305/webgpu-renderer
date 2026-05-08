// Minimal ImportMeta augmentation so import.meta.env.DEV is typed without
// depending on vite/client. Vite replaces DEV with true/false at build time;
// tsup leaves the expression for the consumer bundler to resolve.
interface ImportMeta {
  readonly env: {
    readonly DEV: boolean;
  };
}
