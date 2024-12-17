"use client";

import { useDarkMode } from "@/context/DarkModeContext"; 

// This will help for the dark/light theme toggle button also helps toggle the theme of the page globally 

const ToggleButton = ({ onValueChange }: { onValueChange?: (value: boolean) => void }) => {
  // Access the global dark mode state and toggle function
  const { darkMode, toggleDarkMode } = useDarkMode();

  const handleClick = () => {
    toggleDarkMode();

    if (onValueChange) {
      onValueChange(!darkMode); 
    }
  };

  return (
    <div className="fixed bottom-4 right-4">
      <button
        onClick={handleClick}
        className="bg-white dark:bg-black text-black dark:text-white font-bold py-2 px-4 rounded-full shadow-lg"
      >
        {darkMode ? "Light Mode" : "Dark Mode"}
      </button>
    </div>
  );
};

export default ToggleButton;
