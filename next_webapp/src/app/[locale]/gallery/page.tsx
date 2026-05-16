"use client";

import { useState, useEffect, useCallback } from "react";
import Image from "next/image";
import { useTranslations } from "next-intl";
import { ChevronLeft, ChevronRight } from "lucide-react";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";
import Pagination from "@/components/Pagination";

const GLOW_COLORS = [
  "#22c55e",
  "#3b82f6",
  "#a855f7",
  "#ec4899",
  "#f97316",
  "#14b8a6",
  "#eab308",
] as const;

const ITEMS_PER_PAGE = 21;

type ApiImage = {
  id: number;
  title: string;
  img_url: string;
  created_at: string;
};

export default function GalleryPage() {
  const t = useTranslations("common");

  const [images, setImages] = useState<ApiImage[]>([]);
  const [loading, setLoading] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);

  const lightbox = lightboxIndex !== null ? images[lightboxIndex] : null;

  const fetchImages = useCallback(async (page: number) => {
    setLoading(true);
    try {
      const qs = new URLSearchParams({
        page: String(page),
        pageSize: String(ITEMS_PER_PAGE),
      });
      const res = await fetch(`/api/home/gallery?${qs}`);
      const json = await res.json();
      if (json.success) {
        setImages(json.data ?? []);
        setTotal(json.pagination?.total ?? 0);
        setTotalPages(json.pagination?.totalPages ?? 1);
      }
    } catch (e) {
      console.error("Failed to fetch gallery:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchImages(currentPage);
  }, [currentPage, fetchImages]);

  const closeLightbox = useCallback(() => setLightboxIndex(null), []);

  const goToPreviousImage = useCallback(() => {
    setLightboxIndex((prev) => {
      if (prev === null) return prev;
      return prev === 0 ? images.length - 1 : prev - 1;
    });
  }, [images.length]);

  const goToNextImage = useCallback(() => {
    setLightboxIndex((prev) => {
      if (prev === null) return prev;
      return prev === images.length - 1 ? 0 : prev + 1;
    });
  }, [images.length]);

  useEffect(() => {
    if (!lightbox) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeLightbox();
      if (e.key === "ArrowLeft") goToPreviousImage();
      if (e.key === "ArrowRight") goToNextImage();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [lightbox, closeLightbox, goToPreviousImage, goToNextImage]);

  useEffect(() => {
    document.body.style.overflow = lightbox ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [lightbox]);

  return (
    <>
      <style>{`
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(20px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .anim-fade-up { animation: fadeUp 0.5s ease-out both; }
      `}</style>

      <div className="min-h-screen flex flex-col bg-white dark:bg-[#0f1117]">
        <Header />

        {/* ── Hero ────────────────────────────────────────────── */}
        <section className="relative pt-20 pb-20 px-4 text-center overflow-hidden bg-gradient-to-br from-green-700 via-green-600 to-green-500 dark:from-green-900 dark:via-green-800 dark:to-green-700">
          <div className="absolute -top-16 -left-16 w-64 h-64 bg-white/10 rounded-full blur-2xl pointer-events-none" />
          <div className="absolute -bottom-12 -right-12 w-80 h-80 bg-white/10 rounded-full blur-2xl pointer-events-none" />

          <div className="relative max-w-3xl mx-auto">
            <span
              className="anim-fade-up inline-block px-4 py-1 mb-4 text-xs font-semibold uppercase tracking-widest text-green-100 bg-white/20 rounded-full"
              style={{ animationDelay: "0.1s" }}
            >
              {t("Melbourne Open Data")}
            </span>

            <h1
              className="anim-fade-up text-5xl sm:text-6xl text-white mb-5 drop-shadow-sm"
              style={{
                animationDelay: "0.2s",
                fontWeight: 900,
                fontFamily: "'Barlow Condensed', sans-serif",
                letterSpacing: "-0.02em",
                textShadow: "2px 2px 8px rgba(0,0,0,0.35)",
              }}
            >
              {t("Gallery")}
            </h1>

            <p
              className="anim-fade-up text-lg text-green-100 leading-relaxed max-w-2xl mx-auto"
              style={{ animationDelay: "0.3s" }}
            >
              {t("gallery_subtitle")}
            </p>
          </div>

          <div className="absolute bottom-0 left-0 right-0 overflow-hidden leading-none">
            <svg
              viewBox="0 0 1440 40"
              xmlns="http://www.w3.org/2000/svg"
              className="block w-full fill-white dark:fill-[#0f1117]"
            >
              <path d="M0,20 C360,40 1080,0 1440,20 L1440,40 L0,40 Z" />
            </svg>
          </div>
        </section>

        {/* ── Main grid ───────────────────────────────────────── */}
        <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-10">
          {!loading && (
            <p className="text-sm text-gray-400 dark:text-gray-500 mb-6">
              {t("showing_images", { count: images.length })} of {total}
            </p>
          )}

          {loading ? (
            <div className="flex justify-center py-32">
              <div className="h-10 w-10 animate-spin rounded-full border-4 border-green-500 border-t-transparent" />
            </div>
          ) : images.length === 0 ? (
            <div className="flex min-h-[250px] items-center justify-center rounded-2xl border border-dashed border-gray-300 bg-gray-50 px-6 py-12 text-center dark:border-gray-700 dark:bg-[#242424]">
              <p className="text-base font-medium text-gray-500 dark:text-gray-400">
                No images available at the moment.
              </p>
            </div>
          ) : (
            <>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                {images.map((img, idx) => {
                  const glowColor = GLOW_COLORS[idx % GLOW_COLORS.length];
                  return (
                    <button
                      key={img.id}
                      onClick={() => setLightboxIndex(idx)}
                      onMouseEnter={(e) =>
                        (e.currentTarget.style.boxShadow = `0 0 0 2px ${glowColor}, 0 0 14px 4px ${glowColor}44`)
                      }
                      onMouseLeave={(e) =>
                        (e.currentTarget.style.boxShadow = "")
                      }
                      className="group relative overflow-hidden rounded-2xl bg-gray-100 dark:bg-gray-800 shadow-md transition-transform duration-200 hover:-translate-y-0.5 text-left w-full focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2"
                    >
                      <div className="relative h-64 overflow-hidden">
                        <Image
                          src={img.img_url}
                          alt={img.title}
                          fill
                          sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
                          loading="lazy"
                          className="object-cover transition-transform duration-300 group-hover:scale-105"
                        />
                        <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col justify-end p-5">
                          <h3 className="text-white font-bold text-lg leading-tight">
                            {img.title}
                          </h3>
                        </div>
                      </div>

                      <div className="px-4 py-3 bg-white dark:bg-gray-900 border-t border-gray-100 dark:border-gray-700">
                        <h3 className="text-gray-800 dark:text-white font-semibold text-sm truncate">
                          {img.title}
                        </h3>
                        <p className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">
                          {new Date(img.created_at).toLocaleDateString("en-AU", {
                            day: "2-digit",
                            month: "short",
                            year: "numeric",
                          })}
                        </p>
                      </div>
                    </button>
                  );
                })}
              </div>

              <Pagination
                page={currentPage}
                totalPages={totalPages}
                onPageChange={setCurrentPage}
              />
            </>
          )}
        </main>

        <Footer />

        {/* ── Lightbox ────────────────────────────────────────── */}
        {lightbox && (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-sm"
            onClick={closeLightbox}
            role="dialog"
            aria-modal="true"
            aria-label={lightbox.title}
          >
            <button
              onClick={closeLightbox}
              className="absolute top-4 right-4 z-20 flex items-center justify-center w-10 h-10 rounded-full bg-white/10 hover:bg-white/20 text-white transition-colors focus:outline-none focus:ring-2 focus:ring-white"
              aria-label="Close"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="w-5 h-5"
              >
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>

            <button
              onClick={(e) => { e.stopPropagation(); goToPreviousImage(); }}
              className="absolute left-4 top-1/2 z-20 flex h-11 w-11 -translate-y-1/2 items-center justify-center rounded-full bg-white/10 text-white hover:bg-white/20 focus:outline-none focus:ring-2 focus:ring-white"
              aria-label="Previous image"
            >
              <ChevronLeft size={24} />
            </button>

            <button
              onClick={(e) => { e.stopPropagation(); goToNextImage(); }}
              className="absolute right-4 top-1/2 z-20 flex h-11 w-11 -translate-y-1/2 items-center justify-center rounded-full bg-white/10 text-white hover:bg-white/20 focus:outline-none focus:ring-2 focus:ring-white"
              aria-label="Next image"
            >
              <ChevronRight size={24} />
            </button>

            <div
              className="relative flex flex-col items-center max-w-4xl w-full mx-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="relative w-full max-h-[75vh] rounded-2xl overflow-hidden shadow-2xl">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={lightbox.img_url}
                  alt={lightbox.title}
                  className="w-full h-auto max-h-[75vh] object-contain bg-black"
                />
              </div>

              <div className="mt-4 text-center px-4">
                <h2 className="text-white font-bold text-xl mb-1">
                  {lightbox.title}
                </h2>
                <p className="text-gray-400 text-sm">
                  {new Date(lightbox.created_at).toLocaleDateString("en-AU", {
                    day: "2-digit",
                    month: "long",
                    year: "numeric",
                  })}
                </p>
                {lightboxIndex !== null && (
                  <p className="mt-2 text-xs text-gray-400">
                    {lightboxIndex + 1} / {images.length}
                  </p>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
