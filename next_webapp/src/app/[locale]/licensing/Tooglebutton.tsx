"use client";

import { useState, useEffect } from "react";

interface ToggleButtonProps {
  onValueChange: (value: boolean) => void;
}

const Tooglebutton: React.FC<ToggleButtonProps> = ({ onValueChange }) => {
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    const savedTheme = localStorage.getItem("theme");
    if (savedTheme === "dark") {
      setIsDark(true);
      document.documentElement.classList.add("dark");
      onValueChange(true);
    }
  }, []);

  const handleToggle = () => {
    const newValue = !isDark;
    setIsDark(newValue);
    onValueChange(newValue);
    localStorage.setItem("theme", newValue ? "dark" : "light");

    if (newValue) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  };

  return (
    <button
      onClick={handleToggle}
      className="px-4 py-2 rounded-full shadow-md border border-gray-400 bg-white dark:bg-gray-800 text-black dark:text-white transition duration-300"
    >
      {isDark ? "Light" : "Dark"}
    </button>
  );
};

export default Tooglebutton;
