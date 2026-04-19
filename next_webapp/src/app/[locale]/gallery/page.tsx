"use client";

import { useState, useEffect, useCallback } from "react";
import Image from "next/image";
import { useTranslations } from "next-intl";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";

const CATEGORIES = ["All", "Landmarks", "Environment", "Technology", "Sustainability", "Society"] as const;
type Category = (typeof CATEGORIES)[number];

const GLOW_COLORS = [
  "#22c55e",
  "#3b82f6",
  "#a855f7",
  "#ec4899",
  "#f97316",
  "#14b8a6",
  "#eab308",
] as const;

interface GalleryImage {
  src: string;
  titleKey: string;
  caption: string;
  category: Exclude<Category, "All">;
  glowColor: string;
}

const IMAGES: GalleryImage[] = [
  // Landmarks — iconic Melbourne built environment
  {
    src: "/img/mel.jpg",
    titleKey: "img_federation_square_title",
    caption: "Princes Bridge and the Yarra River frame Melbourne's iconic civic precinct at the heart of the city",
    category: "Landmarks",
    glowColor: GLOW_COLORS[0],
  },
  {
    src: "/img/aboutpic2.jpg",
    titleKey: "img_melbourne_central_title",
    caption: "Melbourne's modern CBD skyline towers above the vibrant shopping and cultural precincts of the city centre",
    category: "Landmarks",
    glowColor: GLOW_COLORS[1],
  },
  {
    src: "/img/queenVictoria.avif",
    titleKey: "img_queen_victoria_market_title",
    caption: "Australia's largest open-air market, operating since 1878",
    category: "Landmarks",
    glowColor: GLOW_COLORS[2],
  },
  {
    src: "/img/melbourne-city.jpg",
    titleKey: "img_docklands_precinct_title",
    caption: "Melbourne's illuminated waterfront at dusk, where the Yarra River meets the Docklands entertainment precinct",
    category: "Landmarks",
    glowColor: GLOW_COLORS[3],
  },
  {
    src: "/img/sliderimg2.jpg",
    titleKey: "img_melbourne_city_skyline_title",
    caption: "The Shrine of Remembrance stands before Melbourne's gleaming CBD skyline in a sweeping aerial panorama",
    category: "Landmarks",
    glowColor: GLOW_COLORS[4],
  },
  {
    src: "/img/sliderimg1.jpg",
    titleKey: "img_flinders_street_station_title",
    caption: "Melbourne's iconic heritage railway station, a beloved city landmark since 1905",
    category: "Landmarks",
    glowColor: GLOW_COLORS[5],
  },
  {
    src: "/img/melbourne-city1.jpg",
    titleKey: "img_melbourne_after_dark_title",
    caption: "The city's lights illuminate Melbourne's vibrant nighttime landscape",
    category: "Landmarks",
    glowColor: GLOW_COLORS[2],
  },
  // Environment — green spaces, biodiversity, climate
  {
    src: "/img/royalBotanic.jpg",
    titleKey: "img_guilfoyles_volcano_title",
    caption: "The restored Victorian-era reservoir at the Royal Botanic Gardens, featuring concentric red-ochre earthen terraces and Indigenous plantings",
    category: "Environment",
    glowColor: GLOW_COLORS[6],
  },
  {
    src: "/img/bio_corridor.jpg",
    titleKey: "img_biodiversity_corridor_title",
    caption: "Ecological connectivity networks supporting urban wildlife",
    category: "Environment",
    glowColor: GLOW_COLORS[0],
  },
  {
    src: "/img/climate_change.jpg",
    titleKey: "img_climate_change_impact_title",
    caption: "Visualising Melbourne's changing climate patterns over time",
    category: "Environment",
    glowColor: GLOW_COLORS[1],
  },
  {
    src: "/img/tree_planting.jpeg",
    titleKey: "img_urban_tree_planting_title",
    caption: "Greening programs expanding the city's tree canopy cover",
    category: "Environment",
    glowColor: GLOW_COLORS[2],
  },
  {
    src: "/img/unique_insects.jpg",
    titleKey: "img_urban_insect_life_title",
    caption: "Documenting Melbourne's unique urban insect populations",
    category: "Environment",
    glowColor: GLOW_COLORS[3],
  },
  {
    src: "/img/urban_climate.jpg",
    titleKey: "img_urban_microclimate_title",
    caption: "Monitoring microclimates across Melbourne's neighbourhoods",
    category: "Environment",
    glowColor: GLOW_COLORS[4],
  },
  // Technology — data, analytics, smart infrastructure
  {
    src: "/img/smart-city.jpg",
    titleKey: "img_smart_city_infrastructure_title",
    caption: "Sensor networks powering Melbourne's data-driven decisions",
    category: "Technology",
    glowColor: GLOW_COLORS[5],
  },
  {
    src: "/img/AI_fire.jpg",
    titleKey: "img_fire_risk_detection_title",
    caption: "AI-assisted fire risk monitoring using open environmental data",
    category: "Technology",
    glowColor: GLOW_COLORS[6],
  },
  {
    src: "/img/heat_island.png",
    titleKey: "img_urban_heat_island_title",
    caption: "Mapping heat islands across the city to guide cooling strategies",
    category: "Technology",
    glowColor: GLOW_COLORS[0],
  },
  {
    src: "/img/heat_map.png",
    titleKey: "img_geospatial_heat_map_title",
    caption: "Heat map visualisations derived from Melbourne's open datasets",
    category: "Technology",
    glowColor: GLOW_COLORS[1],
  },
  {
    src: "/img/Oil-Supply.jpg",
    titleKey: "img_oil_gas_supply_chain_title",
    caption: "Analysing energy supply chain data through Melbourne's open datasets",
    category: "Technology",
    glowColor: GLOW_COLORS[2],
  },
  // Sustainability — mobility, waste, energy, EVs
  {
    src: "/img/sustainable_mobility.jpg",
    titleKey: "img_sustainable_mobility_title",
    caption: "Open data insights driving cleaner transport alternatives",
    category: "Sustainability",
    glowColor: GLOW_COLORS[3],
  },
  {
    src: "/img/waste_efficiency.jpg",
    titleKey: "img_waste_management_efficiency_title",
    caption: "Optimising Melbourne's waste systems through open data",
    category: "Sustainability",
    glowColor: GLOW_COLORS[4],
  },
  {
    src: "/img/ev-banner.png",
    titleKey: "img_electric_vehicle_rollout_title",
    caption: "Tracking EV charging infrastructure growth across Melbourne",
    category: "Sustainability",
    glowColor: GLOW_COLORS[5],
  },
  // Society — education, community, research
  {
    src: "/img/education.jpg",
    titleKey: "img_education_open_data_title",
    caption: "Open data insights improving educational outcomes across the city",
    category: "Society",
    glowColor: GLOW_COLORS[6],
  },
  {
    src: "/img/social_indicator.jpg",
    titleKey: "img_social_wellbeing_indicators_title",
    caption: "Measuring community wellbeing through Melbourne's open data",
    category: "Society",
    glowColor: GLOW_COLORS[0],
  },
  {
    src: "/img/biotech.jpeg",
    titleKey: "img_biotechnology_research_title",
    caption: "Cutting-edge biotech research enabled by open data access",
    category: "Society",
    glowColor: GLOW_COLORS[1],
  },
];

export default function GalleryPage() {
  const t = useTranslations("common");
  const [activeCategory, setActiveCategory] = useState<Category>("All");
  const [lightbox, setLightbox] = useState<GalleryImage | null>(null);

  const closeLightbox = useCallback(() => setLightbox(null), []);

  // Escape key closes the lightbox
  useEffect(() => {
    if (!lightbox) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeLightbox();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [lightbox, closeLightbox]);

  // Prevent body scroll while lightbox is open
  useEffect(() => {
    document.body.style.overflow = lightbox ? "hidden" : "";
    return () => { document.body.style.overflow = ""; };
  }, [lightbox]);

  const filtered =
    activeCategory === "All"
      ? IMAGES
      : IMAGES.filter((img) => img.category === activeCategory);

  return (
    <>
      {/* ── CSS Keyframes ──────────────────────────────────────────── */}
      <style>{`
        /* 1. Hero title fade-up entrance */
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(30px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .anim-fade-up {
          animation: fadeUp 0.8s ease-out both;
        }

        /* 2. Filter button shimmer sweep */
        @keyframes shimmerSweep {
          0%   { transform: translateX(-200%) skewX(-20deg); }
          100% { transform: translateX(600%)  skewX(-20deg); }
        }
        .filter-btn {
          position: relative;
          overflow: hidden;
          border: 1px solid rgba(255, 255, 255, 0.08);
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
          transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .filter-btn::after {
          content: '';
          position: absolute;
          inset: 0;
          left: -60%;
          width: 40%;
          background: linear-gradient(to right, transparent, rgba(255,255,255,0.35), transparent);
          transform: skewX(-20deg);
          pointer-events: none;
        }
        .filter-btn:hover::after {
          animation: shimmerSweep 0.55s ease forwards;
        }
        .filter-btn:hover {
          transform: scale(1.05);
          box-shadow: 0 0 0 1.5px #22c55e, 0 0 14px 3px rgba(34, 197, 94, 0.4);
        }

        /* 4. Hero background shimmer */
        @keyframes heroShimmer {
          0%   { left: -100%; }
          100% { left: 100%; }
        }
        .hero-shimmer {
          position: absolute;
          top: 0;
          bottom: 0;
          width: 40%;
          background: linear-gradient(
            105deg,
            transparent 30%,
            rgba(255, 255, 255, 0.12) 50%,
            transparent 70%
          );
          animation: heroShimmer 6s linear infinite;
          pointer-events: none;
        }

        /* Hero glassmorphism card — infinite shimmer sweep */
        .hero-card-shimmer {
          position: absolute;
          inset: 0;
          background: linear-gradient(105deg, transparent 0%, rgba(255,255,255,0.28) 50%, transparent 100%);
          transform: translateX(-200%) skewX(-20deg);
          pointer-events: none;
          animation: shimmerSweep 4s ease-in-out infinite;
        }
      `}</style>

      <div className="min-h-screen flex flex-col bg-white dark:bg-[#0f1117]">
        <Header />

        {/* ── Hero Section ─────────────────────────────────────────── */}
        <section className="relative pt-20 pb-20 px-4 text-center overflow-hidden bg-gradient-to-br from-green-700 via-green-600 to-green-500 dark:from-green-900 dark:via-green-800 dark:to-green-700">
          {/* 4. Hero shimmer sweep overlay */}
          <div className="hero-shimmer" />

          {/* Decorative blurred circles */}
          <div className="absolute -top-16 -left-16 w-64 h-64 bg-white/10 rounded-full blur-3xl pointer-events-none" />
          <div className="absolute -bottom-12 -right-12 w-80 h-80 bg-white/10 rounded-full blur-3xl pointer-events-none" />

          {/* 1. Staggered fade-up hero content */}
          <div className="relative max-w-3xl mx-auto">
            <span
              className="anim-fade-up inline-block px-4 py-1 mb-4 text-xs font-semibold uppercase tracking-widest text-green-100 bg-white/20 rounded-full backdrop-blur-sm"
              style={{ animationDelay: "0.1s" }}
            >
              {t("Melbourne Open Data")}
            </span>
            <h1
              className="anim-fade-up text-5xl sm:text-6xl text-white mb-5 drop-shadow-sm"
              style={{
                animationDelay: "0.3s",
                fontWeight: 900,
                fontStyle: "normal",
                fontFamily: "'Barlow Condensed', sans-serif",
                letterSpacing: "-0.02em",
                textShadow: "2px 2px 8px rgba(0,0,0,0.4)",
              }}
            >
              {t("Gallery")}
            </h1>
            <p
              className="anim-fade-up text-lg text-green-100 leading-relaxed max-w-2xl mx-auto"
              style={{ animationDelay: "0.5s" }}
            >
              {t("gallery_subtitle")}
            </p>
          </div>

          {/* Wave divider */}
          <div className="absolute bottom-0 left-0 right-0 overflow-hidden leading-none">
            <svg viewBox="0 0 1440 40" xmlns="http://www.w3.org/2000/svg" className="block w-full fill-white dark:fill-[#0f1117]">
              <path d="M0,20 C360,40 1080,0 1440,20 L1440,40 L0,40 Z" />
            </svg>
          </div>
        </section>

        {/* ── Category Filter Bar ───────────────────────────────────── */}
        <div className="sticky top-16 z-30 bg-white/90 dark:bg-[#0f1117]/90 backdrop-blur-md border-b border-gray-100 dark:border-gray-800">
          <div className="max-w-7xl mx-auto px-4 py-3 overflow-x-auto">
            {/* 2. Filter buttons with shimmer + glow */}
            <div className="flex gap-2 whitespace-nowrap w-max min-w-full pb-0.5">
              {CATEGORIES.map((cat) => (
                <button
                  key={cat}
                  onClick={() => setActiveCategory(cat)}
                  className={`filter-btn px-4 py-1.5 rounded-full text-sm font-medium focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-1 ${
                    activeCategory === cat
                      ? "bg-green-600 text-white"
                      : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:text-green-700 dark:hover:text-green-400"
                  }`}
                >
                  {t(`cat_${cat.toLowerCase()}`)}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* ── Image Grid ────────────────────────────────────────────── */}
        <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-10">
          <p className="text-sm text-gray-400 dark:text-gray-500 mb-6">
            {t("showing_images", { count: filtered.length })}
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {filtered.map((img, idx) => (
              <button
                key={`${img.src}-${idx}`}
                onClick={() => setLightbox(img)}
                onMouseEnter={(e) => (e.currentTarget.style.boxShadow = `0 0 0 2px ${img.glowColor}, 0 0 18px 5px ${img.glowColor}55`)}
                onMouseLeave={(e) => (e.currentTarget.style.boxShadow = "")}
                className="group relative overflow-hidden rounded-2xl bg-gray-100 dark:bg-gray-800 shadow-md transition-all duration-300 hover:-translate-y-1 text-left w-full focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2"
              >
                {/* Fixed-height image */}
                <div className="relative h-64 overflow-hidden">
                  <Image
                    src={img.src}
                    alt={t(img.titleKey)}
                    fill
                    sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
                    className="object-cover transition-transform duration-500 group-hover:scale-110"
                  />

                  {/* Hover overlay */}
                  <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col justify-end p-5">
                    <span className="text-xs font-bold uppercase tracking-widest text-green-400 mb-1.5">
                      {t(`cat_${img.category.toLowerCase()}`)}
                    </span>
                    <h3 className="text-white font-bold text-lg leading-tight mb-1">
                      {t(img.titleKey)}
                    </h3>
                    <p className="text-gray-200 text-sm leading-snug line-clamp-2">
                      {img.caption}
                    </p>
                  </div>
                </div>

                {/* Card footer */}
                <div className="px-4 py-3 bg-white dark:bg-gray-900 border-t border-gray-100 dark:border-gray-700">
                  <h3 className="text-gray-800 dark:text-white font-semibold text-sm truncate">
                    {t(img.titleKey)}
                  </h3>
                  <span className="text-xs text-green-600 dark:text-green-400 font-medium">
                    {t(`cat_${img.category.toLowerCase()}`)}
                  </span>
                </div>
              </button>
            ))}
          </div>

          {filtered.length === 0 && (
            <div className="text-center py-24">
              <p className="text-gray-400 dark:text-gray-600 text-lg">
                {t("no_images")}
              </p>
            </div>
          )}
        </main>

        <Footer />

        {/* ── Lightbox Modal ────────────────────────────────────────── */}
        {lightbox && (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-sm"
            onClick={closeLightbox}
            role="dialog"
            aria-modal="true"
            aria-label={t(lightbox.titleKey)}
          >
            {/* Close button */}
            <button
              onClick={closeLightbox}
              className="absolute top-4 right-4 z-10 flex items-center justify-center w-10 h-10 rounded-full bg-white/10 hover:bg-white/20 text-white transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-white"
              aria-label="Close"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>

            {/* Image + caption */}
            <div
              className="relative flex flex-col items-center max-w-4xl w-full mx-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="relative w-full max-h-[75vh] rounded-2xl overflow-hidden shadow-2xl">
                <Image
                  src={lightbox.src}
                  alt={t(lightbox.titleKey)}
                  width={1200}
                  height={800}
                  className="w-full h-auto max-h-[75vh] object-contain bg-black"
                  priority
                />
              </div>
              <div className="mt-4 text-center px-4">
                <h2 className="text-white font-bold text-xl mb-1">{t(lightbox.titleKey)}</h2>
                <p className="text-gray-300 text-sm">{lightbox.caption}</p>
                <span className="inline-block mt-2 px-3 py-0.5 text-xs font-semibold uppercase tracking-wider text-green-400 bg-green-900/40 rounded-full">
                  {t(`cat_${lightbox.category.toLowerCase()}`)}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
