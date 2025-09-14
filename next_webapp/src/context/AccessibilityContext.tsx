'use client'
import React, { createContext, useContext, useState, useEffect } from 'react';

type AccessibilitySettings = {
  darkMode: boolean;
  highContrast: boolean;
  fontSize: string;
  fontFamily: string;
  toggleDarkMode: () => void;
  toggleHighContrast: () => void;
  setFontSize: (size: string) => void;
  setFontFamily: (font: string) => void;
};

const AccessibilityContext = createContext<AccessibilitySettings | undefined>(undefined);

export const AccessibilityProvider = ({ children }: { children: React.ReactNode }) => {
  const [darkMode, setDarkMode] = useState(false);
  const [highContrast, setHighContrast] = useState(false);
  const [fontSize, setFontSize] = useState("medium");
  const [fontFamily, setFontFamily] = useState("sans-serif");

  // Load from localStorage
  useEffect(() => {
    const stored = localStorage.getItem("accessibility-settings");
    if (stored) {
      const parsed = JSON.parse(stored);
      setDarkMode(parsed.darkMode);
      setHighContrast(parsed.highContrast);
      setFontSize(parsed.fontSize);
      setFontFamily(parsed.fontFamily);
    }
  }, []);

  // Save to localStorage whenever settings change
  useEffect(() => {
    localStorage.setItem(
      "accessibility-settings",
      JSON.stringify({ darkMode, highContrast, fontSize, fontFamily })
    );
  }, [darkMode, highContrast, fontSize, fontFamily]);

  return (
    <AccessibilityContext.Provider
      value={{
        darkMode,
        highContrast,
        fontSize,
        fontFamily,
        toggleDarkMode: () => setDarkMode((prev) => !prev),
        toggleHighContrast: () => setHighContrast((prev) => !prev),
        setFontSize,
        setFontFamily,
      }}
    >
      <div
        className={`${darkMode ? "dark" : ""} ${highContrast ? "contrast" : ""}`}
        style={{ fontFamily, fontSize: fontSize === "small" ? "14px" : fontSize === "large" ? "18px" : "16px" }}
      >
        {children}
      </div>
    </AccessibilityContext.Provider>
  );
};

export const useAccessibility = () => {
  const ctx = useContext(AccessibilityContext);
  if (!ctx) throw new Error("useAccessibility must be used inside AccessibilityProvider");
  return ctx;
};
