'use client'
import { useState, useEffect, useCallback } from 'react';

type Theme = 'light' | 'dark';

/**
 * Custom hook to manage light/dark theme.
 * - Reads initial value synchronously from localStorage (lazy initializer) to
 *   avoid the flash caused by defaulting to 'light' then correcting after mount.
 * - Toggles the 'dark' class on <html>.
 * - Persists choice to localStorage.
 */
export function useTheme(): { theme: Theme; toggleTheme: () => void } {
  // Lazy initializer: reads localStorage synchronously on the client so the
  // very first render already has the correct theme — no effect-based correction
  // needed, which previously caused the dark class to be removed momentarily.
  const [theme, setTheme] = useState<Theme>(() => {
    if (typeof window === 'undefined') return 'light';
    const saved = localStorage.getItem('theme') as Theme | null;
    if (saved) return saved;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  });

  // Whenever theme changes, update <html> class and persist
  useEffect(() => {
    const root = window.document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));
  }, []);

  return { theme, toggleTheme };
}
