"use client";
import React from "react";
import { Link } from "@/i18n-navigation";
import { useCases } from "@/utils/data";

const categoryStyles: Record<string, { badge: string; border: string }> = {
  "business-and-economy":        { badge: "bg-blue-100 text-blue-700",   border: "hover:border-blue-400" },
  "community-and-social-impact": { badge: "bg-pink-100 text-pink-700",   border: "hover:border-pink-400" },
  "education-and-teaching":      { badge: "bg-yellow-100 text-yellow-700", border: "hover:border-yellow-400" },
  "environmental-sustainability":{ badge: "bg-green-100 text-green-700", border: "hover:border-green-400" },
  "safety-and-security":         { badge: "bg-red-100 text-red-700",     border: "hover:border-red-400" },
  "tourism-and-hospitality":     { badge: "bg-orange-100 text-orange-700", border: "hover:border-orange-400" },
  "transport-and-mobility":      { badge: "bg-purple-100 text-purple-700", border: "hover:border-purple-400" },
  "urban-planing-and-development":{ badge: "bg-indigo-100 text-indigo-700", border: "hover:border-indigo-400" },
  "helth-and-wellbeign":         { badge: "bg-teal-100 text-teal-700",   border: "hover:border-teal-400" },
};

const formatBadgeLabel = (id: string): string => {
  return id
    .split("-")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};

const Insights: React.FC = () => {
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
          Explore Our Use Cases
        </h2>
        <p className="text-gray-600 dark:text-gray-300 text-sm md:text-base">
          Discover how our innovative solutions transform industries worldwide.
        </p>
      </div>

      <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {useCases.map((useCase) => {
          const style = categoryStyles[useCase.id] || {
            badge: "bg-gray-100 text-gray-700",
            border: "hover:border-gray-400",
          };

          return (
            <Link
              key={useCase.id}
              href={`/usecases/${useCase.id}`}
              className={`bg-gray-50 dark:bg-[#37474F] hover:bg-gray-100 dark:hover:bg-[#455A64] rounded-2xl shadow-md hover:shadow-2xl border-2 border-transparent ${style.border} transition-all duration-300 flex flex-col group hover:-translate-y-1`}
            >
              {/* Image */}
              <div className="overflow-hidden rounded-t-2xl">
                <img
                  src={useCase.image}
                  alt={useCase.title}
                  className="w-full h-44 object-cover group-hover:scale-105 transition-transform duration-300"
                />
              </div>

              {/* Card Body */}
              <div className="p-4 flex flex-col flex-grow">
                {/* Category Badge */}
                <span
                  className={`self-start text-xs font-semibold px-3 py-1 rounded-full mb-3 ${style.badge}`}
                >
                  {formatBadgeLabel(useCase.id)}
                </span>

                {/* Title */}
                <h3 className="text-lg font-bold mb-2 leading-snug">
                  {useCase.title}
                </h3>

                {/* Description */}
                <p className="text-gray-600 dark:text-gray-300 text-sm flex-grow leading-relaxed">
                  {useCase.description}
                </p>

                {/* Button */}
                <span className="mt-5 bg-green-500 group-hover:bg-green-600 group-hover:shadow-lg text-white py-2 px-4 rounded-xl text-sm font-semibold text-center transition-all duration-300 flex items-center justify-center gap-2">
                  View Details
                  <span className="group-hover:translate-x-1 transition-transform duration-300">→</span>
                </span>
              </div>
            </Link>
          );
        })}
      </div>
    </section>
  );
};

export default Insights;