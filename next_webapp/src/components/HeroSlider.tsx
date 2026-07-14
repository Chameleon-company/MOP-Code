"use client";
import Image from "next/image";
import { motion, AnimatePresence } from "framer-motion";
import { useRef } from "react";
export const HERO_SLIDES = [
  { src: "/img/mainImage.png",    alt: "Melbourne city overview" },
  { src: "/img/mel.jpg",          alt: "Melbourne aerial view" },
  { src: "/img/sliderimg1.jpg",   alt: "Melbourne cityscape" },
  { src: "/img/sliderimg2.jpg",   alt: "Melbourne Central" },
  { src: "/img/royalBotanic.jpg", alt: "Royal Botanic Gardens Melbourne" },
] as const;

// ─── Component ─────────────────────────────────────────────────────────────────

interface HeroSliderProps {
  /** Zero-based index of the slide to display (controlled by parent). */
  currentIndex: number;
  /** Called when the user swipes left — parent should advance to next slide. */
  onNext: () => void;
  /** Called when the user swipes right — parent should go to previous slide. */
  onPrev: () => void;
}

export default function HeroSlider({ currentIndex, onNext, onPrev }: HeroSliderProps) {
  const slide = HERO_SLIDES[currentIndex];

  // ── Touch / swipe detection ────────────────────────────────────────────────
  // Record the X position where the finger first touched the screen.
  const touchStartX = useRef<number>(0);

  const handleTouchStart = (e: React.TouchEvent) => {
    touchStartX.current = e.touches[0].clientX;
  };

  const handleTouchEnd = (e: React.TouchEvent) => {
    const delta = e.changedTouches[0].clientX - touchStartX.current;
    // Only trigger if the swipe was at least 50px — avoids accidental taps
    if (delta < -50) onNext();     // swipe left  → advance
    else if (delta > 50) onPrev(); // swipe right → go back
  };

  return (
    <div
      className="hero-image-container"
      onTouchStart={handleTouchStart}
      onTouchEnd={handleTouchEnd}
    >
      <AnimatePresence
        // "sync" = old image fades out while new one fades in (crossfade)
        mode="sync"
      >
        <motion.div
          key={currentIndex}
          // Fade in from transparent
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          // Fade out when replaced by the next slide
          exit={{ opacity: 0 }}
          transition={{ duration: 0.9, ease: "easeInOut" }}
          // Must be absolute + fill parent so images can use Next.js fill layout
          style={{ position: "absolute", inset: 0, overflow: "hidden" }}
        >
          <Image
            src={slide.src}
            alt={slide.alt}
            // fill = image stretches to cover parent; parent must be positioned
            fill
            // Only the first slide gets a <link rel="preload"> for performance
            priority={currentIndex === 0}
            // Responsive sizes hint: tells Next.js the image is always full-width
            // so it picks the smallest adequate srcset on mobile devices
            sizes="(max-width: 480px) 100vw, (max-width: 768px) 100vw, 100vw"
            // hero-slide-img applies the Ken Burns keyframe defined in Dashboard.tsx
            className="hero-slide-img"
            style={{ objectFit: "cover" }}
          />
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
