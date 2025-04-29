'use client'

import { useEffect, useState } from 'react';
import { HiMoon, HiSun } from 'react-icons/hi';
 
const ThemeToggle = () => {
  const [isDark, setIsDark] = useState(false);
 
  useEffect(() => {
    const storedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const isDarkMode = storedTheme === 'dark' || (!storedTheme && prefersDark);
 
    setIsDark(isDarkMode);
    document.documentElement.classList.toggle('dark', isDarkMode);
  }, []);
 
  const toggleTheme = () => {
    const newTheme = !isDark;
    setIsDark(newTheme);
    document.documentElement.classList.toggle('dark', newTheme);
    localStorage.setItem('theme', newTheme ? 'dark' : 'light');
  };
 
  return (
    <button
      onClick={toggleTheme}
      className="ml-4 text-xl text-green-600 hover:text-green-800"
      aria-label="Toggle Dark Mode"
    >
      {isDark ? <HiSun /> : <HiMoon />}
    </button>
  );
};
export default ThemeToggle;