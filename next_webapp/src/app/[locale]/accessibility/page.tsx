'use client'
import React from "react";
import { useAccessibility } from "@/context/AccessibilityContext";
import Header from "@/components/Header";
import Footer from "@/components/Footer";

const AccessibilityPage = () => {
  const {
    darkMode,
    highContrast,
    fontSize,
    fontFamily,
    toggleDarkMode,
    toggleHighContrast,
    setFontSize,
    setFontFamily,
  } = useAccessibility();

  return (
    <div
      style={{
        backgroundImage: "url('/img/accessibility.png')",
        backgroundSize: "cover",
        backgroundPosition: "center",
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <Header />

      {/* Main Content */}
      <main className="flex-grow flex items-center justify-center p-10">
        <div className="bg-white dark:bg-gray-900 bg-opacity-95 dark:bg-opacity-95 rounded-2xl shadow-xl max-w-5xl w-full grid grid-cols-1 md:grid-cols-2 gap-10 p-8">
          
          {/* Left Navigation */}
          <div>
            <h2 className="text-xl font-bold mb-4">Accessibility Options</h2>
            <div className="flex flex-col gap-4">
              <button className="p-3 rounded-lg border hover:bg-gray-100 dark:hover:bg-gray-800">Text & Visual</button>
              <button className="p-3 rounded-lg border hover:bg-gray-100 dark:hover:bg-gray-800">Keyboard & Navigation</button>
              <button className="p-3 rounded-lg border hover:bg-gray-100 dark:hover:bg-gray-800">Audio & Captions</button>
              <button className="p-3 rounded-lg border hover:bg-gray-100 dark:hover:bg-gray-800">Language & Multilingual</button>
              <button className="p-3 rounded-lg border hover:bg-gray-100 dark:hover:bg-gray-800">Feedback & Help</button>
            </div>
          </div>

          {/* Right Settings */}
          <div>
            <h2 className="text-xl font-bold mb-4">Adjust Settings</h2>

            {/* Font Style */}
            <label className="block font-medium mb-2">Font Style</label>
            <select
              value={fontFamily}
              onChange={(e) => setFontFamily(e.target.value)}
              className="p-2 w-full rounded-lg border"
            >
              <option value="sans-serif">Sans-serif</option>
              <option value="serif">Serif</option>
              <option value="dyslexia">Dyslexia-Friendly</option>
            </select>

            {/* Font Size */}
            <label className="block font-medium mt-4 mb-2">Font Size</label>
            <div className="flex gap-3">
              <button onClick={() => setFontSize("small")} className="px-4 py-2 rounded bg-blue-500 text-white">Small</button>
              <button onClick={() => setFontSize("medium")} className="px-4 py-2 rounded bg-blue-500 text-white">Medium</button>
              <button onClick={() => setFontSize("large")} className="px-4 py-2 rounded bg-blue-500 text-white">Large</button>
            </div>

            {/* Toggles */}
            <div className="mt-6 space-y-3">
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={highContrast} onChange={toggleHighContrast} />
                High Contrast
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" checked={darkMode} onChange={toggleDarkMode} />
                Dark Mode
              </label>
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default AccessibilityPage;
