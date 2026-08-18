const isBrowser = () => typeof window !== "undefined";

export const storage = {
  getItem(key: string): string | null {
    if (!isBrowser()) return null;

    return window.localStorage.getItem(key);
  },

  setItem(key: string, value: string): void {
    if (!isBrowser()) return;

    window.localStorage.setItem(key, value);
  },

  removeItem(key: string): void {
    if (!isBrowser()) return;

    window.localStorage.removeItem(key);
  },

  clear(): void {
    if (!isBrowser()) return;

    window.localStorage.clear();
  },
};
