"use client";



// edits for use case studies
import { Link, useRouter } from "@/i18n-navigation";

import Image from "next/image";
import secondimage from "../../public/img/second_image.png";
import HeroSlider, { HERO_SLIDES } from "@/components/HeroSlider";
import { useTranslations } from "next-intl";
import { CaseStudy, CATEGORY, SEARCH_MODE, SearchParams } from "@/app/types";
import { useEffect, useState, useRef } from "react";
import {
	ArrowRight,
	Play,
	ChevronDown,
	Search,
	X,
} from "lucide-react";
import CityMetricCard, { CityMetric } from "@/components/CityMetricCard";
import { Users, Car, Trees, Home, DollarSign, Heart } from "lucide-react";


const cityMetrics: CityMetric[] = [
	{
		id: "1",
		title: "Population",
		value: "2.3M",
		change: 2.5,
		icon: <Users size={20} className="text-blue-700 dark:text-blue-300" />,
		category: "population",
	},
	{
		id: "2",
		title: "Public Transport",
		value: "78%",
		change: 5.2,
		icon: <Car size={20} className="text-purple-700 dark:text-purple-300" />,
		category: "transportation",
	},
	{
		id: "3",
		title: "Green Spaces",
		value: "32%",
		change: -1.2,
		icon: <Trees size={20} className="text-green-700 dark:text-green-300" />,
		category: "environment",
	},
	{
		id: "4",
		title: "Housing Affordability",
		value: "64%",
		change: 3.1,
		icon: <Home size={20} className="text-amber-700 dark:text-amber-300" />,
		category: "housing",
	},
	{
		id: "5",
		title: "Median Income",
		value: "$65,420",
		change: 4.7,
		icon: (
			<DollarSign size={20} className="text-indigo-700 dark:text-indigo-300" />
		),
		category: "economy",
	},
	{
		id: "6",
		title: "Life Expectancy",
		value: "81.2 yrs",
		change: 0.8,
		icon: <Heart size={20} className="text-pink-700 dark:text-pink-300" />,
		category: "health",
	},
];




const style = `
.main-container {
  background: white;
  color: #263238;
  display: flex;
  flex-direction: column;
  margin-bottom: 0;
}
.dark .main-container {
  background: #263238;
  color: white;
  flex: 1;
}

/* Enhanced Hero Section — overflow visible so the search preview can extend below;
   background images stay clipped via .hero-image-container */
.hero-section {
  width: 100%;
  height: 90vh;
  min-height: 700px;
  overflow: visible;
  position: relative;
  z-index: 20;
  display: flex;
  align-items: center;
  justify-content: center;
}
.hero-image-container {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 1;
  overflow: hidden;
}
.hero-image-container::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background:
    linear-gradient(to bottom, rgba(0,0,0,0.15) 0%, transparent 35%, rgba(0,0,0,0.65) 100%),
    linear-gradient(to right, rgba(0,0,0,0.55) 0%, rgba(0,0,0,0.1) 50%, rgba(0,0,0,0.55) 100%);
  z-index: 2;
  transition: background 0.5s ease;
}
.dark .hero-image-container::before {
  background: rgba(10,15,18,0.25);
}
.dark .hero-image-container img {
  filter: blur(2px) brightness(0.75);
}
.hero-image-container img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.hero-content {
  position: relative;
  z-index: 3;
  text-align: center;
  color: white;
  max-width: 900px;
  padding: 0 2rem;
}
.hero-title {
  font-size: 3.5rem;
  font-weight: 800;
  margin-bottom: 1.5rem;
  line-height: 1.2;
  text-shadow:
    1px 1px 0 rgba(0,0,0,0.4),
    2px 2px 0 rgba(0,0,0,0.35),
    3px 3px 0 rgba(0,0,0,0.3),
    4px 4px 0 rgba(0,0,0,0.25),
    5px 5px 0 rgba(0,0,0,0.2),
    6px 6px 0 rgba(0,0,0,0.15),
    6px 6px 15px rgba(0,0,0,0.4);
  animation: fadeInUp 0.8s ease-out 0.1s both;
}
.hero-subtitle {
  font-size: 1.5rem;
  margin-bottom: 2.5rem;
  font-weight: 400;
  max-width: 700px;
  margin-left: auto;
  margin-right: auto;
  color: rgba(255, 255, 255, 0.95);
  text-shadow: 0 1px 8px rgba(0, 0, 0, 0.6);
  animation: fadeInUp 0.8s ease-out 0.45s both;
}
.hero-buttons {
  display: flex;
  gap: 1rem;
  justify-content: center;
  margin-bottom: 2rem;
  animation: fadeInUp 0.8s ease-out 0.75s both;
}
.hero-button {
  padding: 1rem 2rem;
  border-radius: 50px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  transition: all 0.3s ease;
  cursor: pointer;
}
.hero-button.primary {
  background: linear-gradient(135deg, #10B981 0%, #059669 100%);
  color: white;
  position: relative;
  overflow: hidden;
  box-shadow: 0 4px 15px rgba(16, 185, 129, 0.25);
}
.hero-button.primary::after {
  content: '';
  position: absolute;
  top: -50%;
  left: -60%;
  width: 35%;
  height: 200%;
  background: rgba(255,255,255,0.25);
  transform: skewX(-20deg);
  animation: shimmer 3s 1.5s infinite;
}
.hero-button.primary:hover {
  background: linear-gradient(135deg, #059669 0%, #047857 100%);
  transform: translateY(-3px);
  box-shadow: 0 14px 30px rgba(16, 185, 129, 0.4);
}
.hero-button.primary:active {
  transform: translateY(-1px);
}
.hero-button.secondary {
  background: rgba(255,255,255,0.12);
  color: white;
  border: 2px solid rgba(255,255,255,0.75);
  backdrop-filter: blur(6px);
}
.hero-button.secondary:hover {
  background: rgba(255,255,255,0.22);
  border-color: white;
  transform: translateY(-3px);
  box-shadow: 0 8px 20px rgba(0,0,0,0.2);
}
.hero-button.secondary:active {
  transform: translateY(-1px);
}
.typing-cursor {
  display: inline-block;
  width: 3px;
  height: 0.85em;
  background: white;
  margin-left: 4px;
  vertical-align: middle;
  border-radius: 1px;
  animation: blink 1s step-end infinite;
}
.typing-cursor.done {
  animation: none;
  opacity: 0;
  transition: opacity 0.5s ease 0.8s;
}

/* Search Container */
.search-container {
  background: rgba(255, 255, 255, 0.15);
  backdrop-filter: blur(10px);
  border-radius: 50px;
  padding: 0.5rem;
  margin: 0 auto;
  max-width: 600px;
  display: flex;
  align-items: center;
  border: 1px solid rgba(255, 255, 255, 0.2);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);

  z-index: 10;
}
.search-input {
  flex: 1;
  background: transparent;
  border: none;
  padding: 0.8rem 1.5rem;
  color: white;
  font-size: 1rem;
  outline: none;
}
.search-input::placeholder {
  color: rgba(255, 255, 255, 0.7);
}
.search-options {
  display: flex;
  gap: 0.5rem;
  padding: 0 1rem;
 
}
.search-option {
  background: rgba(255, 255, 255, 0.2);
  border: none;
  border-radius: 20px;
  padding: 0.4rem 0.8rem;
  color: white;
  font-size: 0.8rem;
  cursor: pointer;
  transition: all 0.3s ease;
}
.search-option:hover, .search-option.active {
  background: #10B981;
}
.search-button {
  background: #10B981;
  border: none;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.3s ease;
}
.search-button:hover {
  background: #059669;
  transform: rotate(15deg);
}
.clear-search {
  background: rgba(255, 255, 255, 0.2);
  border: none;
  border-radius: 50%;
  width: 30px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  margin-right: 0.5rem;
  transition: all 0.3s ease;
}
.clear-search:hover {
  background: rgba(255, 255, 255, 0.3);
}

.search-results {
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  width: 100%;
  max-width: min(100%, 640px);
  margin-top: 0.75rem;
  background: white;
  border-radius: 16px;
  border: 1px solid rgba(0, 0, 0, 0.08);
  box-shadow:
    0 4px 6px -1px rgba(0, 0, 0, 0.08),
    0 20px 40px -12px rgba(0, 0, 0, 0.18);
  z-index: 100;
  overflow: hidden;
  text-align: left;
}
.dark .search-results {
  background: #2d3e47;
  color: #eceff1;
  border-color: rgba(255, 255, 255, 0.12);
  box-shadow:
    0 4px 6px -1px rgba(0, 0, 0, 0.35),
    0 24px 48px -12px rgba(0, 0, 0, 0.45);
}
.search-results-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  padding: 0.65rem 1rem;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #546e7a;
  background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
  border-bottom: 1px solid #e2e8f0;
}
.dark .search-results-header {
  color: #90a4ae;
  background: linear-gradient(180deg, #37474f 0%, #2d3e47 100%);
  border-bottom-color: rgba(255, 255, 255, 0.1);
}
.search-results-count {
  font-size: 0.7rem;
  font-weight: 800;
  letter-spacing: 0.04em;
  padding: 0.2rem 0.55rem;
  border-radius: 9999px;
  background: #ecfdf5;
  color: #047857;
}
.dark .search-results-count {
  background: rgba(16, 185, 129, 0.2);
  color: #6ee7b7;
}
.search-results-scroll {
  max-height: clamp(200px, 42vh, 380px);
  overflow-y: auto;
  overscroll-behavior: contain;
  -webkit-overflow-scrolling: touch;
  padding-bottom: 10px;
}
.search-results-scroll::-webkit-scrollbar {
  width: 8px;
}
.search-results-scroll::-webkit-scrollbar-thumb {
  background: rgba(0, 0, 0, 0.15);
  border-radius: 9999px;
}
.dark .search-results-scroll::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.18);
}
button.search-result-item {
  display: block;
  width: 100%;
  margin: 0;
  padding: 0.85rem 1rem;
  text-align: left;
  font: inherit;
  color: inherit;
  cursor: pointer;
  border: none;
  border-bottom: 1px solid #eceff1;
  background: transparent;
  transition: background 0.15s ease;
}
.dark button.search-result-item {
  border-bottom-color: rgba(255, 255, 255, 0.08);
}
button.search-result-item:last-of-type {
  border-bottom: none;
}
button.search-result-item:hover {
  background: #f8fafc;
}
.dark button.search-result-item:hover {
  background: rgba(255, 255, 255, 0.06);
}
button.search-result-item:focus-visible {
  outline: 2px solid #10b981;
  outline-offset: -2px;
}
.search-result-title {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  margin: 0 0 0.35rem 0;
  font-size: 0.98rem;
  font-weight: 600;
  line-height: 1.35;
  color: #0f172a;
}
.dark .search-result-title {
  color: #f1f5f9;
}
.search-result-snippet {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  margin: 0;
  font-size: 0.82rem;
  line-height: 1.45;
  color: #64748b;
}
.dark .search-result-snippet {
  color: #94a3b8;
}
.search-result-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  margin-top: 0.5rem;
}
.search-result-tag {
  font-size: 0.65rem;
  font-weight: 600;
  padding: 0.15rem 0.45rem;
  border-radius: 9999px;
  background: rgba(16, 185, 129, 0.12);
  color: #047857;
  max-width: 140px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.dark .search-result-tag {
  background: rgba(16, 185, 129, 0.18);
  color: #a7f3d0;
}
.search-results-hint {
  padding: 0.5rem 1rem 0.65rem;
  font-size: 0.68rem;
  color: #90a4ae;
  text-align: center;
  border-top: 1px solid #eceff1;
  background: #fafafa;
}
.dark .search-results-hint {
  color: #78909c;
  border-top-color: rgba(255, 255, 255, 0.08);
  background: rgba(0, 0, 0, 0.15);
}
.no-results {
  padding: 1.75rem 1.25rem;
  text-align: center;
  color: #78909c;
}
.dark .no-results {
  color: #b0bec5;
}
.no-results p {
  margin: 0.5rem 0 0 0;
  font-size: 0.9rem;
}
.no-results .loading-spinner {
  margin-bottom: 0.25rem;
}

.scroll-indicator {
  position: absolute;
  bottom: 30px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 3;
  animation: bounce 2s infinite;
  color: white;
}
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
@keyframes bounce {
  0%, 20%, 50%, 80%, 100% {
    transform: translateY(0) translateX(-50%);
  }
  40% {
    transform: translateY(-20px) translateX(-50%);
  }
  60% {
    transform: translateY(-10px) translateX(-50%);
  }
}
@keyframes shimmer {
  0% { left: -60%; }
  100% { left: 130%; }
}
@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}
@media (max-width: 768px) {
  .hero-title {
    font-size: 2.5rem;
  }
  .hero-subtitle {
    font-size: 1.2rem;
  }
  .hero-buttons {
    flex-direction: column;
    align-items: center;
  }
  .search-container {
    flex-direction: column;
    border-radius: 12px;
    padding: 1rem;
  }
  .search-input {
    width: 100%;
    margin-bottom: 0.5rem;
  }
  .search-options {
    width: 100%;
    justify-content: center;
    margin-bottom: 0.5rem;
  }
  .search-option {
    flex: 1;
    text-align: center;
  }
  .search-results {
    width: calc(100vw - 1.5rem);
    max-width: none;
    margin-top: 0.65rem;
    border-radius: 14px;
  }
  .search-results-scroll {
    max-height: min(48vh, 320px);
  }

  /* ── Hero slider — mobile overrides ─────────────────────────────────── */

  /* Reduce min-height so the hero does not overflow short phone screens */
  .hero-section {
    min-height: 580px;
    height: 100svh; /* svh = small viewport height, accounts for mobile browser chrome */
  }

  /* On narrow screens the side gradients waste too much width; simplify to
     a strong bottom fade only so the image subject stays fully visible */
  .hero-image-container::before {
    background: linear-gradient(
      to bottom,
      rgba(0,0,0,0.2) 0%,
      transparent 30%,
      rgba(0,0,0,0.72) 100%
    );
  }

  /* Gentler zoom on mobile — reduces motion and saves battery */
  .hero-slide-img {
    animation: kenBurnsMobile 6s ease-in-out forwards;
  }

  /* Hide the absolutely-positioned desktop dots on mobile */
  .hero-slider-dots--desktop { display: none; }

  /* Mobile dots — pinned to top of hero-section, between nav bar and headline */
  .hero-slider-dots--mobile {
    display: flex;
    position: absolute;
    top: 24px;
    bottom: auto;            /* override the desktop bottom value */
    left: 50%;
    transform: translateX(-50%);
    justify-content: center;
    margin: 0;
    gap: 12px;
  }

  /* Larger visual dot and an expanded invisible tap target via ::after */
  .hero-dot {
    width: 12px;
    height: 12px;
    position: relative;
  }
  .hero-dot::after {
    content: "";
    position: absolute;
    inset: -14px; /* expands tap area to ~40px without changing visual size */
  }
  .hero-dot.active {
    width: 30px;
  }
}

/* ── Hero Slider ─────────────────────────────────────────────────────────────── */

/* Ken Burns: subtle slow-zoom on each active slide.
   Because every slide is a freshly-mounted component (keyed by index),
   this animation automatically restarts for each new image. */
@keyframes kenBurns {
  from { transform: scale(1.0); }
  to   { transform: scale(1.1) translate(-1%, -0.5%); }
}
/* Applied to the Next.js <img> inside each slide motion.div */
.hero-slide-img {
  object-fit: cover;
  animation: kenBurns 6s ease-in-out forwards;
  will-change: transform;
}

/* Subtler zoom used on mobile (applied inside the 768px block below) */
@keyframes kenBurnsMobile {
  from { transform: scale(1.0); }
  to   { transform: scale(1.05); }
}

/* Respect the OS-level "reduce motion" preference — disable Ken Burns entirely */
@media (prefers-reduced-motion: reduce) {
  .hero-slide-img {
    animation: none;
  }
}

/* Dot indicator strip — rendered as a sibling to hero-content at z-index 4 */
.hero-slider-dots {
  position: absolute;
  bottom: 115px; /* ~45px clear gap above the scroll-indicator chevron (bottom:30px + 40px height) */
  left: 50%;
  transform: translateX(-50%);
  z-index: 4;
  display: flex;
  gap: 8px;
  align-items: center;
}
/* Two-class selectors (specificity 0,2,0) beat all single-class rules above,
   regardless of cascade order — this is the authoritative mobile override */
@media (max-width: 768px) {
  .hero-slider-dots.hero-slider-dots--mobile {
    display: flex;  /* beats the single-class display:none defined later */
    top: 24px;
    bottom: auto;
  }
  .hero-slider-dots.hero-slider-dots--desktop {
    display: none;  /* beats the single-class display:flex defined later */
  }
}
/* Individual dot */
.hero-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.45);
  border: 2px solid rgba(255, 255, 255, 0.7);
  cursor: pointer;
  transition: all 0.3s ease;
  padding: 0;
  flex-shrink: 0;
}
/* Active dot stretches into a pill */
.hero-dot.active {
  width: 26px;
  border-radius: 5px;
  background: white;
  border-color: white;
}
.hero-dot:hover:not(.active) {
  background: rgba(255, 255, 255, 0.75);
  transform: scale(1.2);
}

/* Desktop: show the absolutely-positioned dots, hide the inline ones */
.hero-slider-dots--mobile { display: none; }
.hero-slider-dots--desktop { display: flex; }

.our-vision-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  background: #ffffff;
  color: #263238;
  margin: 4rem auto;
  gap: 2rem;
  padding: 2.5rem;
  width: 90%;
  max-width: 1200px;
  box-sizing: border-box;
  border-radius: 1rem;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.our-vision-section:hover {
  transform: translateY(-4px);
  box-shadow: 0 10px 24px rgba(0, 0, 0, 0.12);
}
.dark .our-vision-section {
  background: #324148;
  color: white;
}
@media (min-width: 768px) {
  .our-vision-section {
     flex-direction: row;
    align-items: center;
    
  }
}
.img-div {
  flex: 1;
  width: 100%;
  max-width: 460px;
  border-radius: 0.75rem;
  overflow: hidden;
}
.img-div img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  transition: transform 0.3s ease;
}
.our-vision-section:hover .img-div img {
  transform: scale(1.03);
}
.text-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
.our-vision {
  font-weight: 800;
  font-size: 2.25rem;
  margin: 0;
}
.case-studies-wrapper {
  background-color: #F6F9FC;
  padding: 2rem;
}
.dark .case-studies-wrapper {
  background-color: rgb(46, 38, 38);
}
.recent-case-studies {
  margin: 0 auto;
  padding: 0 2rem;
  max-width: 1200px;
  text-align: center;
  color: #263238;
}
.dark .recent-case-studies {
  color: white;
}
.recent-case-studies p {
  font-size: 1rem;
  margin-bottom: 3rem;
  width: 50%;
  margin-left: auto;
  margin-right: auto;
}

/* Loading states */
.loading-spinner {
  border: 3px solid #f3f3f3;
  border-top: 3px solid #10B981;
  border-radius: 50%;
  width: 30px;
  height: 30px;
  animation: spin 1s linear infinite;
  margin: 0 auto;
}
@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* Debug info panel */
.debug-panel {
  background: #f8f9fa;
  border: 1px solid #e9ecef;
  border-radius: 8px;
  padding: 1rem;
  margin: 1rem 0;
  font-family: monospace;
  font-size: 0.8rem;
  max-height: 200px;
  overflow-y: auto;
}
.text-container p {
  font-size: 1rem;
  line-height: 1.8;
  color: #455a64;
  margin: 0;
}

.dark .text-container p {
  color: #d9e3e8;
}
.dark .debug-panel {
  background: #2d3748;
  border-color: #4a5568;
}
`;


/** Hero search: explicit modes so Title / Content / Tag always map to the API correctly */
const HERO_SEARCH_MODES: { value: SEARCH_MODE; label: string }[] = [
	{ value: SEARCH_MODE.TITLE, label: "Title" },
	{ value: SEARCH_MODE.CONTENT, label: "Content" },
	{ value: SEARCH_MODE.TAG, label: "Tag" },
];

const Dashboard = () => {

  //edits for use case studies
  const router = useRouter();

	const t = useTranslations("common");
	const t_hero = useTranslations("hero");
	const heroTitle = t_hero("hero-top");
	const [displayedTitle, setDisplayedTitle] = useState("");
	const [isTypingDone, setIsTypingDone] = useState(false);
	const [filteredCaseStudies, setFilteredCaseStudies] = useState<CaseStudy[]>(
		[]
	);
	const [searchTerm, setSearchTerm] = useState("");
	const [searchMode, setSearchMode] = useState<SEARCH_MODE>(SEARCH_MODE.TITLE);
	const [category, setCategory] = useState<CATEGORY>(CATEGORY.ALL);
	const [showSearchResults, setShowSearchResults] = useState(false);
	const [isSearching, setIsSearching] = useState(false);
	const [recentUseCases, setRecentUseCases] = useState<any[]>([]);
	const [recentLoading, setRecentLoading] = useState(true);
	const [homeCategories, setHomeCategories] = useState<any[]>([]);

	// ── Hero slider state ────────────────────────────────────────────────────────
	// currentSlide: index of the visible background image
	// sliderTimerKey: incrementing this value resets the auto-advance interval,
	//   which gives a better UX when the user manually picks a slide via a dot.
	const [currentSlide, setCurrentSlide] = useState(0);
	const [sliderTimerKey, setSliderTimerKey] = useState(0);

	// Auto-advance background every 5 s; restarts whenever sliderTimerKey changes
	useEffect(() => {
		const timer = setInterval(() => {
			setCurrentSlide((prev) => (prev + 1) % HERO_SLIDES.length);
		}, 5000);
		return () => clearInterval(timer);
	}, [sliderTimerKey]);

	// Jump to a specific slide and reset the auto-advance countdown
	const goToSlide = (index: number) => {
		setCurrentSlide(index);
		setSliderTimerKey((k) => k + 1);
	};

	// Convenience helpers used by swipe gestures (HeroSlider) and dots
	const handleNext = () => goToSlide((currentSlide + 1) % HERO_SLIDES.length);
	const handlePrev = () => goToSlide((currentSlide - 1 + HERO_SLIDES.length) % HERO_SLIDES.length);

	// Hero search: Title / Content / Tag (see HERO_SEARCH_MODES)
	const searchContainerRef = useRef<HTMLDivElement>(null);

	// Live hero search: debounce 350ms, fetch top 5 matching usecases for dropdown
	useEffect(() => {
		if (!searchTerm.trim()) {
			setFilteredCaseStudies([]);
			setShowSearchResults(false);
			return;
		}
		const timer = setTimeout(async () => {
			setIsSearching(true);
			try {
				const params = new URLSearchParams({ pageSize: "5" });
				if (searchMode === SEARCH_MODE.TAG) {
					params.set("tag_name", searchTerm.trim());
				} else {
					params.set("search", searchTerm.trim());
					params.set("search_by", searchMode);
				}
				const res = await fetch(`/api/usecases?${params}`);
				const json = await res.json();
				if (json.success) {
					setFilteredCaseStudies(
						(json.data || []).map((u: any) => ({
							id: u.id,
							title: u.title,
							description: u.description ?? "",
							tags: (u.tags || []).map((t: any) => (typeof t === "string" ? t : t.name)),
							htmlFile: "",
						}))
					);
					setShowSearchResults(true);
				}
			} catch {
				// silently ignore
			} finally {
				setIsSearching(false);
			}
		}, 350);
		return () => clearTimeout(timer);
	}, [searchTerm, searchMode]);

	useEffect(() => {
		fetch("/api/usecases/recent")
			.then((r) => r.json())
			.then((json) => { if (json.success) setRecentUseCases(json.data || []); })
			.catch(() => {})
			.finally(() => setRecentLoading(false));
	}, []);

	useEffect(() => {
		fetch("/api/home/categories")
			.then((r) => r.json())
			.then((json) => { if (json.success) setHomeCategories(json.data || []); })
			.catch(() => {});
	}, []);

	// Add click outside handler
	useEffect(() => {
		const handleClickOutside = (event: MouseEvent) => {
			if (
				searchContainerRef.current &&
				!searchContainerRef.current.contains(event.target as Node)
			) {
				setShowSearchResults(false);
			}
		};

		document.addEventListener("mousedown", handleClickOutside);
		return () => {
			document.removeEventListener("mousedown", handleClickOutside);
		};
	}, []);

	// Typing effect for hero headline
	useEffect(() => {
		let i = 0;
		setDisplayedTitle("");
		setIsTypingDone(false);
		const delay = setTimeout(() => {
			const timer = setInterval(() => {
				if (i < heroTitle.length) {
					setDisplayedTitle(heroTitle.slice(0, i + 1));
					i++;
				} else {
					setIsTypingDone(true);
					clearInterval(timer);
				}
			}, 25);
			return () => clearInterval(timer);
		}, 100);
		return () => clearTimeout(delay);
	}, [heroTitle]);

	const mapApiRowToCaseStudy = (u: Record<string, unknown>): CaseStudy => {
		const rawTags = Array.isArray(u.tags) ? u.tags : [];
		const tags = rawTags.map((t) =>
			typeof t === "string" ? t : String((t as { name?: string })?.name ?? ""),
		);

		return {
			id: Number(u.id),
			title: String(u.title ?? ""),
			description: typeof u.description === "string" ? u.description : "",
			tags,
			htmlFile: "",
			category: CATEGORY.ALL,
		};
	};

	const fetchUsecasesSearch = async (
		term: string,
		mode: SEARCH_MODE,
	): Promise<CaseStudy[]> => {
		const trimmed = term.trim();
		const params = new URLSearchParams();
		params.set("pageSize", "20");
		params.set("page", "1");

		if (trimmed) {
			if (mode === SEARCH_MODE.TAG) {
				params.set("tag_name", trimmed);
			} else if (mode === SEARCH_MODE.TITLE) {
				params.set("q", trimmed);
				params.set("search_by", "title");
			} else {
				/* Content: match legacy behaviour — search title, description, and notebook body */
				params.set("q", trimmed);
				params.set("search_by", "all");
			}
		}

		const response = await fetch(`/api/usecases?${params.toString()}`);

		if (!response.ok) {
			throw new Error(`HTTP error! status: ${response.status}`);
		}

		const json = (await response.json()) as {
			success?: boolean;
			data?: Record<string, unknown>[];
			error?: string;
		};

		if (!json.success) {
			throw new Error(json.error || "Search failed");
		}

		return (json.data ?? []).map(mapApiRowToCaseStudy);
	};

	const handleSearch = async (
		searchTermArg: string,
		searchModeArg: SEARCH_MODE,
		_category: CATEGORY,
	) => {
		setIsSearching(true);
		try {
			const studies = await fetchUsecasesSearch(searchTermArg, searchModeArg);
			setFilteredCaseStudies(studies);
		} catch (error) {
			console.error("Search error:", error);
			setFilteredCaseStudies([]);
		} finally {
			setIsSearching(false);
		}
	};

	const handleCaseStudyClick = (study: CaseStudy) => {
		setShowSearchResults(false);
		router.push(`/usecases?search=${encodeURIComponent(study.title)}&search_by=title`);
	};

	const scrollToContent = () => {
		document.querySelector(".our-vision-section")?.scrollIntoView({
			behavior: "smooth",
		});
	};

	const handleSearchSubmit = (e: React.FormEvent) => {
		e.preventDefault();
		if (!searchTerm.trim()) return;
		const params = new URLSearchParams();
		if (searchMode === SEARCH_MODE.TAG) {
			params.set("tag_name", searchTerm.trim());
		} else {
			params.set("search", searchTerm.trim());
			params.set("search_by", searchMode);
		}
		router.push(`/usecases?${params.toString()}`);
	};

	const clearSearch = () => {
		setSearchTerm("");
		setShowSearchResults(false);
		setFilteredCaseStudies([]);
	};

	return (
		<>

			<style dangerouslySetInnerHTML={{ __html: style }} />
			<div className="main-wrapper bg-white dark:bg-[#263238] text-black dark:text-white min-h-screen">
				<div className="main-container">
					<section className="hero-section">
						{/* Background image slider: HeroSlider reuses .hero-image-container so
						     the gradient overlay (::before) and dark-mode filter keep working. */}
						<HeroSlider currentIndex={currentSlide} onNext={handleNext} onPrev={handlePrev} />

{/* Mobile slide dots — absolutely positioned at top of hero-section,
                          visually between the nav bar and the headline */}
						<div className="hero-slider-dots hero-slider-dots--mobile" aria-label="Slide indicators">
							{HERO_SLIDES.map((_, idx) => (
								<button
									key={idx}
									className={`hero-dot${currentSlide === idx ? " active" : ""}`}
									onClick={() => goToSlide(idx)}
									aria-label={`Go to slide ${idx + 1}`}
								/>
							))}
						</div>

{/* hero content section */}
						<div className="hero-content">
							<h1 className="hero-title">
								{displayedTitle}
								<span className={`typing-cursor${isTypingDone ? " done" : ""}`} aria-hidden="true" />
							</h1>
							<p className="hero-subtitle">{t_hero("hero-sub")}</p>

							<div className="hero-buttons">
								<Link href="/usecases" className="hero-button primary">
									{t_hero("exploreCaseStudies")} <ArrowRight size={20} />
								</Link>
								<a
									href="https://youtu.be/D-H-nCrQDZo"
									target="_blank"
									rel="noopener noreferrer"
									className="hero-button secondary inline-flex items-center gap-2"
								>
									<Play size={20} fill="white" />
									{t_hero("watchVideo")}
								</a>
							</div>

							{/* Search Container with ref for outside click detection */}
							<div
								ref={searchContainerRef}
								style={{ position: "relative", width: "100%" }}
							>
								<form
									onSubmit={handleSearchSubmit}
									className="search-container"
								>
									<input
										type="text"
										placeholder={t_hero("place_holder")}
										className="search-input"
										value={searchTerm}
										onChange={(e) => setSearchTerm(e.target.value)}
										onFocus={() => setShowSearchResults(true)}
									/>
									{searchTerm && (
										<button
											type="button"
											className="clear-search"
											onClick={clearSearch}
										>
											<X size={16} />
										</button>
									)}
									<div className="search-options">
										{HERO_SEARCH_MODES.map(({ value, label }) => (
											<button
												key={value}
												type="button"
												className={`search-option ${
													searchMode === value ? "active" : ""
												}`}
												onClick={() => {
													setSearchMode(value);
													if (searchTerm.trim()) {
														void handleSearch(
															searchTerm.trim(),
															value,
															category,
														);
													}
												}}
											>
												{label}
											</button>
										))}
									</div>
									<button type="submit" className="search-button">
										<Search size={20} />
									</button>
								</form>

								{showSearchResults && (
									<div
										className="search-results"
										role="region"
										aria-label="Search results"
										aria-busy={isSearching}
									>
										{!isSearching && filteredCaseStudies.length > 0 ? (
											<div className="search-results-header">
												<span>Use cases</span>
												<span className="search-results-count">
													{filteredCaseStudies.length}
												</span>
											</div>
										) : null}
										<div className="search-results-scroll">
											{isSearching ? (
												<div className="no-results">
													<div className="loading-spinner" />
													<p>Searching…</p>
												</div>
											) : filteredCaseStudies.length > 0 ? (
												filteredCaseStudies.map((study) => (
													<button
														key={study.id}
														type="button"
														className="search-result-item"
														onClick={() => handleCaseStudyClick(study)}
													>
														<span className="search-result-title">
															{study.title}
														</span>
														<span className="search-result-snippet">
															{study.description || ""}
														</span>
														{study.tags && study.tags.length > 0 ? (
															<div className="search-result-tags">
																{study.tags.slice(0, 4).map((tag, i) => (
																	<span
																		key={`${study.id}-tag-${i}`}
																		className="search-result-tag"
																		title={
																			typeof tag === "string"
																				? tag
																				: String(tag)
																		}
																	>
																		{typeof tag === "string"
																			? tag
																			: String(tag)}
																	</span>
																))}
															</div>
														) : null}
													</button>
												))
											) : (
												<div className="no-results">
													<p>No use cases match that search.</p>
													<p style={{ fontSize: "0.82rem", opacity: 0.85 }}>
														Try another keyword or switch Title / Content /
														Tag.
													</p>
												</div>
											)}
										</div>
										{!isSearching && filteredCaseStudies.length > 0 ? (
											<div className="search-results-hint">
												Click a result to open the use case
											</div>
										) : null}
									</div>
								)}
							</div>
						</div>

						{/* Desktop dots — absolutely positioned at hero section bottom.
						     Hidden on mobile; replaced by inline dots inside hero-content. */}
						<div className="hero-slider-dots hero-slider-dots--desktop" aria-label="Slide indicators">
							{HERO_SLIDES.map((_, idx) => (
								<button
									key={idx}
									className={`hero-dot${currentSlide === idx ? " active" : ""}`}
									onClick={() => goToSlide(idx)}
									aria-label={`Go to slide ${idx + 1}`}
								/>
							))}
						</div>

						<div className="scroll-indicator" onClick={scrollToContent}>
							<ChevronDown size={40} />
						</div>
					</section>
					{/* City Metric Solution  */}
					<section className="city-metrics-section relative z-0 bg-gray-50 dark:bg-gray-900 py-12">
						<div className="container mx-auto px-4">
							<h2 className="text-3xl font-bold text-center mb-8 text-gray-800 dark:text-white">
								City Metrics Overview
							</h2>
							<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
								{cityMetrics.map((metric) => (
									<CityMetricCard key={metric.id} metric={metric} />
								))}
							</div>
						</div>
					</section>

{/* Our vision section */}
					<section className="our-vision-section">
						<div className="img-div">
							<Image src={secondimage} alt="Second Image" />
						</div>
						<div className="text-container">
							<h2 className="our-vision">{t("Our Vision")}</h2>
							<p>{t("intro")}</p>
						</div>
					</section>


					<section className="case-studies-wrapper">
						<section className="recent-case-studies">
							<h2 className="text-3xl md:text-4xl font-bold mb-2">
								Recent Use Cases
							</h2>
							<p>{t("p2")}</p>
						</section>

            {/* Recent use cases from backend */}
            {recentLoading ? (
              <div className="flex justify-center py-10 mt-8">
                <div className="loading-spinner" />
              </div>
            ) : (
              <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 mt-8">
                {recentUseCases.map((item) => (
                  <div
                    key={item.id}
                    className="bg-white dark:bg-[#2f4048] rounded-2xl border border-gray-200 dark:border-gray-700 shadow-sm hover:shadow-md transition flex flex-col group overflow-hidden"
                  >
                    <img
                      src={item.cover_img || "/img/biotech.jpeg"}
                      alt={item.title}
                      className="w-full h-40 object-cover group-hover:scale-[1.02] transition-transform"
                    />
                    <div className="p-5 flex flex-col flex-grow">
                      <h3 className="text-lg font-semibold mb-2 text-left">
                        {item.title}
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 text-sm text-left flex-grow line-clamp-2">
                        {item.description}
                      </p>
                      <div className="mt-4 flex flex-wrap gap-2">
                        {(item.tags ?? []).map((tag: { id: number; name: string }) => (
                          <span
                            key={tag.id}
                            className="rounded-full border border-gray-200 dark:border-gray-600 px-3 py-1 text-xs text-gray-600 dark:text-gray-300 bg-gray-50 dark:bg-[#4b5e67]"
                          >
                            {tag.name}
                          </span>
                        ))}
                      </div>
                      <button
                        onClick={() => router.push(`/usecases/${item.id}`)}
                        className="mt-4 bg-green-500 hover:bg-green-600 text-white py-2 px-4 rounded-xl text-sm font-medium text-center"
                      >
                        View Details →
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}

						</section>




				</div>

			</div>


		</>
	);


};

export default Dashboard;