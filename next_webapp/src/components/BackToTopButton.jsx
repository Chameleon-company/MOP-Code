"use client";
import { useEffect, useState } from "react";
import { IoChevronUp } from "react-icons/io5";

export default function BackToTopButton() {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const toggleVisibility = () => {
      setVisible(window.scrollY > 300);
    };

    window.addEventListener("scroll", toggleVisibility);
    return () => window.removeEventListener("scroll", toggleVisibility);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  };

  if (!visible) return null;

  return (
    <button
      onClick={scrollToTop}
      className="fixed bottom-24 right-4 z-[9998] text-3xl text-white bg-green-600 rounded-full p-3 hover:bg-green-700 shadow-lg transition"
      aria-label="Back to top"
      title="Back to top"
    >
      <IoChevronUp />
    </button>
  );
}