"use client";
import React, { useState, useRef, useEffect } from "react";

const LanguageDropdown: React.FC = () => {
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const ref = useRef<HTMLDivElement>(null);

  // Custom hook to handle clicks outside the dropdown
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (ref.current && !ref.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="border-[1px] border-solid border-white text-[#09bd09] font-serif py-3 px-6 mx-3 bg-white rounded-full  text-lg"
      >
        Language
      </button>
      {isOpen && (
        <div className="absolute z-10 bg-white rounded shadow-lg mt-1 w-48">
          <a
            href="#"
            className="block font-serif px-4 py-2 text-lg rounded text-gray-700 hover:bg-gray-100"
          >
            English
          </a>
          <a
            href="#"
            className="block font-serif px-4 py-2 text-lg rounded text-gray-700 hover:bg-gray-100"
          >
            Chinese
          </a>
        </div>
      )}
    </div>
  );
};

export default LanguageDropdown;
