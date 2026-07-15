"use client";
import React, { useState, useEffect } from "react";
import { Link } from "@/i18n-navigation";


const categoryStyles: Record<string, { badge: string; border: string }> = {
  "business-and-economy":          { badge: "bg-blue-100 text-blue-700",    border: "hover:border-blue-400" },
  "community-and-social-impact":   { badge: "bg-pink-100 text-pink-700",    border: "hover:border-pink-400" },
  "education-and-teaching":        { badge: "bg-yellow-100 text-yellow-700", border: "hover:border-yellow-400" },
  "environmental-sustainability":  { badge: "bg-green-100 text-green-700",  border: "hover:border-green-400" },
  "safety-and-security":           { badge: "bg-red-100 text-red-700",      border: "hover:border-red-400" },
  "tourism-and-hospitality":       { badge: "bg-orange-100 text-orange-700", border: "hover:border-orange-400" },
  "transport-and-mobility":        { badge: "bg-purple-100 text-purple-700", border: "hover:border-purple-400" },
  "urban-planing-and-development": { badge: "bg-indigo-100 text-indigo-700", border: "hover:border-indigo-400" },
  "helth-and-wellbeign":           { badge: "bg-teal-100 text-teal-700",    border: "hover:border-teal-400" },
};

function toSlug(name: string): string {
  return name.toLowerCase().replace(/\s+/g, "-").replace(/[^a-z0-9-]/g, "");
}

const Insights: React.FC = () => {
  const [categories, setCategories] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/home/categories")
      .then((r) => r.json())
      .then((json) => { if (json.success) setCategories(json.data || []); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return (
    <section
      className="w-full bg-white dark:bg-[#263238] py-12 px-6 text-black dark:text-white"
      aria-labelledby="usecases-heading"
    >
      <div className="text-center mb-10">
        <h2
          id="usecases-heading"
          className="text-3xl md:text-4xl font-bold mb-2"
        >
          Explore By Categories
        </h2>
        <p className="text-gray-600 dark:text-gray-300 text-sm md:text-base">
          Discover how our innovative solutions transform industries worldwide.
        </p>
      </div>

      {loading ? (
        <div className="flex justify-center py-10">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-green-500 border-t-transparent" />
        </div>
      ) : categories.length === 0 ? (
        <div className="flex min-h-[200px] items-center justify-center rounded-2xl border border-dashed border-gray-300 bg-gray-50 px-6 py-12 text-center dark:border-gray-700 dark:bg-[#37474F]">
          <p className="text-base font-medium text-gray-500 dark:text-gray-400">
            No insights available at the moment.
          </p>
        </div>
      ) : (
        <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {categories.map((cat) => {
            const slug = toSlug(cat.category_name);
            const style = categoryStyles[slug] || {
              badge: "bg-gray-100 text-gray-700",
              border: "hover:border-gray-400",
            };

            return (
              <Link
                key={cat.id}
                href={`/categories/${cat.id}`}
                className={`bg-gray-50 dark:bg-[#37474F] hover:bg-gray-100 dark:hover:bg-[#455A64] rounded-2xl shadow-md hover:shadow-2xl border-2 border-transparent ${style.border} transition-all duration-300 flex flex-col group hover:-translate-y-1`}
              >
                <div className="overflow-hidden rounded-t-2xl">
                  <img
                    src={cat.cover_img || "/img/insights/eco.webp"}
                    alt={cat.category_name}
                    className="w-full h-44 object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                </div>

                <div className="p-4 flex flex-col flex-grow">
                  <span className={`self-start text-xs font-semibold px-3 py-1 rounded-full mb-3 ${style.badge}`}>
                    {cat.category_name}
                  </span>

                  <h3 className="text-lg font-bold mb-2 leading-snug">
                    {cat.category_name}
                  </h3>

                  <p className="text-gray-600 dark:text-gray-300 text-sm flex-grow leading-relaxed">
                    {cat.description || ""}
                  </p>

                  <span className="mt-5 bg-green-500 group-hover:bg-green-600 group-hover:shadow-lg text-white py-2 px-4 rounded-xl text-sm font-semibold text-center transition-all duration-300 flex items-center justify-center gap-2">
                    View Details
                    <span className="group-hover:translate-x-1 transition-transform duration-300">→</span>
                  </span>
                </div>
              </Link>
            );
          })}
        </div>
      )}
    </section>
  );
};

export default Insights;