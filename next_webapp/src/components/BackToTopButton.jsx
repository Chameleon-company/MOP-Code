"use client";
import { useEffect, useState } from "react";
import { IoChevronUp } from "react-icons/io5";

export default function BackToTopButton() {
  const [visible, setVisible] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);

    const toggleVisibility = () => {
      setVisible(window.scrollY > 300);
    };

    window.addEventListener("scroll", toggleVisibility);
    toggleVisibility();

    return () => window.removeEventListener("scroll", toggleVisibility);
  }, []);

  if (!mounted) return null;

  
  return (
    <button
      onClick={() =>
        window.scrollTo({
          top: 0,
          behavior: "smooth",
        })
      }
      aria-label="Back to top"
      title="Back to top"
      className={`fixed right-4 bottom-24 z-[9998] text-3xl text-white bg-green-600 rounded-full p-3 shadow-lg hover:bg-green-700 transition-all duration-300 ease-in-out ${
        visible
          ? "opacity-100 translate-y-0 pointer-events-auto"
          : "opacity-0 translate-y-3 pointer-events-none"
      }`}
    >
      <IoChevronUp />
    </button>
  );
}