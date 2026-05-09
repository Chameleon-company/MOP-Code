"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  BriefcaseBusiness,
  GraduationCap,
  HeartPulse,
  Leaf,
  MapPinned,
  Search,
  ShieldCheck,
  Sparkles,
  Train,
  Users,
  X,
} from "lucide-react";
import Header from "../../../components/Header";
import Footer from "../../../components/Footer";

interface Category {
  id: number;
  name: string;
  slug?: string;
  color?: string;
  icon?: string;
  description?: string;
  image?: string;
}

const getCategoryIcon = (categoryName: string) => {
  switch (categoryName) {
    case "Business and Economy":
      return <BriefcaseBusiness size={16} />;
    case "Community and Social Impact":
      return <Users size={16} />;
    case "Education and Teaching":
      return <GraduationCap size={16} />;
    case "Environmental Sustainability":
      return <Leaf size={16} />;
    case "Health and Wellbeing":
      return <HeartPulse size={16} />;
    case "Safety and Security":
      return <ShieldCheck size={16} />;
    case "Tourism and Hospitality":
      return <MapPinned size={16} />;
    case "Transport and Mobility":
      return <Train size={16} />;
    default:
      return <Sparkles size={16} />;
  }
};

const getFallbackImage = (categoryName: string) => {
  switch (categoryName) {
    case "Business and Economy":
      return "/images/business.jpg";
    case "Community and Social Impact":
      return "/images/community.jpg";
    case "Education and Teaching":
      return "/images/education.jpg";
    case "Environmental Sustainability":
      return "/images/environment.jpg";
    case "Health and Wellbeing":
      return "/images/health.jpg";
    case "Safety and Security":
      return "/images/safety.jpg";
    case "Tourism and Hospitality":
      return "/images/tourism.jpg";
    case "Transport and Mobility":
      return "/images/transport.jpg";
    case "Urban Planning and Development":
      return "/images/urban.jpg";
    default:
      return "/img/biotech.jpeg";
  }
};

const normaliseCategory = (item: any): Category => {
  const name =
    item.name ||
    item.title ||
    item.category_name ||
    item.categoryName ||
    item.heading ||
    "Untitled Category";

  return {
    id: item.id,
    name,
    slug: item.slug || item.category_slug || item.categorySlug,
    color: item.color || item.colour || "#22c55e",
    icon: item.icon,
    description:
      item.description ||
      item.desc ||
      item.short_description ||
      item.shortDescription ||
      "Explore open data content related to this category.",
    image:
      item.image ||
      item.img ||
      item.cover_img ||
      item.coverImage ||
      item.cover_image ||
      item.thumbnail ||
      getFallbackImage(name),
  };
};

export default function CategoriesPage() {
  const [categories, setCategories] = useState<Category[]>([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<Category | null>(
    null
  );
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchCategories();
  }, []);

  const fetchCategories = async () => {
    try {
      const response = await fetch("/api/home/categories");
      const data = await response.json();

      console.log("Home Categories API Response:", data);

      if (data.success && Array.isArray(data.data)) {
        const formattedCategories = data.data.map((item: any) =>
          normaliseCategory(item)
        );

        setCategories(formattedCategories);
      } else {
        setCategories([]);
      }
    } catch (error) {
      console.error("Error fetching home categories:", error);
      setCategories([]);
    } finally {
      setLoading(false);
    }
  };

  const filteredCategories = useMemo(() => {
    return categories.filter((category) =>
      category.name.toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [categories, searchTerm]);

  return (
    <>
      <style>{`
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(20px); }
          to   { opacity: 1; transform: translateY(0); }
        }

        .anim-fade-up {
          animation: fadeUp 0.5s ease-out both;
        }

        .category-card {
          transition: transform 0.25s ease, box-shadow 0.25s ease;
        }

        .category-card:hover {
          transform: translateY(-4px);
          box-shadow: 0 16px 35px rgba(15, 23, 42, 0.12);
        }
      `}</style>

      <div className="min-h-screen flex flex-col bg-[#f5f7f9] dark:bg-[#0f1117]">
        <Header />

        {/* Hero Section */}
        <section className="relative pt-20 pb-20 px-4 text-center overflow-hidden bg-gradient-to-br from-green-700 via-green-600 to-green-500 dark:from-green-900 dark:via-green-800 dark:to-green-700">
          <div className="absolute -top-16 -left-16 w-64 h-64 bg-white/10 rounded-full blur-2xl pointer-events-none" />
          <div className="absolute -bottom-12 -right-12 w-80 h-80 bg-white/10 rounded-full blur-2xl pointer-events-none" />

          <div className="relative max-w-3xl mx-auto">
            <span
              className="anim-fade-up inline-block px-4 py-1 mb-4 text-xs font-semibold uppercase tracking-widest text-green-100 bg-white/20 rounded-full"
              style={{ animationDelay: "0.1s" }}
            >
              Melbourne Open Data
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
              Categories
            </h1>

            <p
              className="anim-fade-up text-lg text-green-100 leading-relaxed max-w-2xl mx-auto"
              style={{ animationDelay: "0.3s" }}
            >
              Browse all smart city categories and explore how Melbourne open
              data is organised across different areas.
            </p>
          </div>

          <div className="absolute bottom-0 left-0 right-0 overflow-hidden leading-none">
            <svg
              viewBox="0 0 1440 40"
              xmlns="http://www.w3.org/2000/svg"
              className="block w-full fill-[#f5f7f9] dark:fill-[#0f1117]"
            >
              <path d="M0,20 C360,40 1080,0 1440,20 L1440,40 L0,40 Z" />
            </svg>
          </div>
        </section>

        <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-10">
          {loading ? (
            <div className="flex min-h-[350px] items-center justify-center">
              <div className="text-center">
                <div className="mx-auto mb-4 h-10 w-10 animate-spin rounded-full border-4 border-green-200 border-t-green-600" />
                <p className="text-gray-500 dark:text-gray-400">
                  Loading categories...
                </p>
              </div>
            </div>
          ) : (
            <>
              {/* Counters */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-5 mb-10">
                <div className="rounded-2xl bg-white dark:bg-gray-900 border border-gray-100 dark:border-gray-800 shadow-md p-5">
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Total Categories
                  </p>
                  <h3 className="mt-2 text-3xl font-black text-gray-900 dark:text-white">
                    {categories.length}
                  </h3>
                </div>

                <div className="rounded-2xl bg-white dark:bg-gray-900 border border-gray-100 dark:border-gray-800 shadow-md p-5">
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Showing Categories
                  </p>
                  <h3 className="mt-2 text-3xl font-black text-gray-900 dark:text-white">
                    {filteredCategories.length}
                  </h3>
                </div>

                <div className="rounded-2xl bg-white dark:bg-gray-900 border border-gray-100 dark:border-gray-800 shadow-md p-5">
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Selected Category
                  </p>
                  <h3 className="mt-2 text-xl font-black text-gray-900 dark:text-white line-clamp-1">
                    {selectedCategory?.name || "None"}
                  </h3>
                </div>
              </div>

              {/* Search */}
              <div className="mb-8 rounded-2xl bg-white dark:bg-gray-900 border border-gray-100 dark:border-gray-800 shadow-md p-5">
                <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
                  Search Categories
                </label>

                <div className="relative">
                  <Search
                    size={18}
                    className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400"
                  />

                  <input
                    type="text"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    placeholder="Search by category name..."
                    className="w-full rounded-full border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-[#242424] px-11 py-3 text-sm text-gray-800 dark:text-white outline-none focus:border-green-500 focus:ring-2 focus:ring-green-500/20"
                  />

                  {searchTerm && (
                    <button
                      onClick={() => setSearchTerm("")}
                      className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-700 dark:hover:text-white"
                      aria-label="Clear search"
                    >
                      <X size={18} />
                    </button>
                  )}
                </div>
              </div>

              {/* Category Gallery Header */}
              <section className="mb-6 rounded-3xl bg-white dark:bg-gray-900 border border-gray-100 dark:border-gray-800 shadow-sm px-6 py-6">
                <p className="mb-2 text-sm font-bold uppercase tracking-[0.35em] text-green-600">
                  Category Gallery
                </p>

                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                  <div>
                    <h2 className="text-3xl font-black text-gray-900 dark:text-white">
                      Tiled showcase of categories
                    </h2>
                    <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
                      Showing {filteredCategories.length} of {categories.length}{" "}
                      categories
                    </p>
                  </div>
                </div>
              </section>

              {/* Category Cards */}
              {filteredCategories.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 items-stretch">
                  {filteredCategories.map((category) => {
                    const isSelected = selectedCategory?.id === category.id;

                    return (
                      <article
                        key={category.id}
                        className={`category-card h-full min-h-[520px] rounded-3xl bg-white dark:bg-gray-900 border overflow-hidden flex flex-col ${
                          isSelected
                            ? "border-green-500 ring-2 ring-green-500/20"
                            : "border-gray-100 dark:border-gray-800"
                        }`}
                      >
                        {/* Image */}
                        <div className="h-[190px] w-full overflow-hidden bg-gray-100 dark:bg-gray-800 shrink-0">
                          <img
                            src={category.image}
                            alt={category.name}
                            className="h-full w-full object-cover transition-transform duration-300 hover:scale-105"
                          />
                        </div>

                        {/* Content */}
                        <div className="p-6 flex flex-col flex-1">
                          <span
                            className="mb-4 inline-flex w-fit items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold"
                            style={{
                              backgroundColor: `${
                                category.color || "#22c55e"
                              }20`,
                              color: category.color || "#22c55e",
                            }}
                          >
                            {getCategoryIcon(category.name)}
                            {category.name}
                          </span>

                          <h3 className="mb-3 text-2xl font-black leading-tight text-gray-900 dark:text-white min-h-[64px] line-clamp-2">
                            {category.name}
                          </h3>

                          <p className="mb-6 text-sm leading-7 text-gray-600 dark:text-gray-300 min-h-[96px] line-clamp-4">
                            {category.description}
                          </p>

                          <button
                            onClick={() => setSelectedCategory(category)}
                            className="mt-auto inline-flex w-full items-center justify-center gap-2 rounded-xl bg-green-500 px-5 py-3 text-sm font-bold text-white transition hover:bg-green-600"
                          >
                            View Details
                            <ArrowRight size={16} />
                          </button>
                        </div>
                      </article>
                    );
                  })}
                </div>
              ) : (
                <div className="flex min-h-[220px] items-center justify-center rounded-2xl border border-dashed border-gray-300 bg-white px-6 py-12 text-center dark:border-gray-700 dark:bg-[#242424]">
                  <p className="text-base font-medium text-gray-500 dark:text-gray-400">
                    No active categories found.
                  </p>
                </div>
              )}
            </>
          )}
        </main>

        <Footer />
      </div>
    </>
  );
}