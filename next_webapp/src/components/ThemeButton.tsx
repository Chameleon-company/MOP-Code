'use client';

import { useState, useLayoutEffect } from 'react';
import { HiMoon, HiSun } from 'react-icons/hi';

function useDarkMode() {
  const [darkMode, setDarkMode] = useState(false);

  useLayoutEffect(() => {
    const userPreference = localStorage.getItem('theme');
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const shouldUseDark = userPreference ? userPreference === 'dark' : systemPrefersDark;

    if (shouldUseDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    setDarkMode(shouldUseDark);
  }, []);

  const toggleDarkMode = () => {
    setDarkMode(prev => {
      const newMode = !prev;
      document.documentElement.classList.toggle('dark', newMode);
      localStorage.setItem('theme', newMode ? 'dark' : 'light');
      return newMode;
    });
  };

  return [darkMode, toggleDarkMode] as const;
}

const ThemeButton = () => {
  const [darkMode, toggleDarkMode] = useDarkMode();

  return (
    <div className="ml-4">
      <button
        onClick={toggleDarkMode}
        aria-label="Switch Theme"
        className="text-xl text-green-600 hover:text-green-800 transition-colors"
      >
        {darkMode ? <HiSun /> : <HiMoon />}
      </button>
    </div>
  );
};

export default ThemeButton;

// 'use client'

// import { useEffect, useState } from 'react';
// import { HiMoon, HiSun } from 'react-icons/hi';

// const ThemeToggle = () => {
//   const [isDark, setIsDark] = useState(false);

//   useEffect(() => {
//     const storedTheme = localStorage.getItem('theme');
//     const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
//     const isDarkMode = storedTheme === 'dark' || (!storedTheme && prefersDark);

//     setIsDark(isDarkMode);
//     document.documentElement.classList.toggle('dark', isDarkMode);
//   }, []);

//   const toggleTheme = () => {
//     const newTheme = !isDark;
//     setIsDark(newTheme);
//     document.documentElement.classList.toggle('dark', newTheme);
//     localStorage.setItem('theme', newTheme ? 'dark' : 'light');
//   };

//   return (
//     <button
//       onClick={toggleTheme}
//       className="ml-4 text-xl text-green-600 hover:text-green-800"
//       aria-label="Toggle Dark Mode"
//     >
//       {isDark ? <HiSun /> : <HiMoon />}
//     </button>
//   );
// };
// export default ThemeToggle;