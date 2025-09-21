"use client";

import React from "react";
import Link from "next/link";
import { notFound, useParams } from "next/navigation";
import usecasesData from "../../../utils/usecases.json";
import Header from "../../../../components/Header";
import Footer from "../../../../components/Footer";
import { useLocale } from "next-intl";

interface UseCase {
  name: string;
  address: string;
}
interface Category {
  category: string;
  usecases: UseCase[];
}

function slugify(text: string) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)+/g, "");
}

export default function CategoryPage() {
  const params = useParams<{ category: string }>();
  const categorySlug = params.category;

  const category = (usecasesData as Category[]).find(
    (cat) => slugify(cat.category) === categorySlug
  );

  if (!category) {
    notFound();
  }

  const locale = useLocale();

  return (
    <div className="flex flex-col min-h-screen font-sans bg-white dark:bg-gray-800 text-black dark:text-white transition-all duration-300">
      <Header />

      <main className="flex-grow max-w-5xl mx-auto px-4 py-8">
        {/* Breadcrumbs */}
        <nav className="mb-6 text-sm text-gray-600 dark:text-gray-300">
          <Link href="/" className="hover:underline">Home</Link> &gt;{" "}
          <Link href={`/${locale}/usecases`} className="hover:underline">Use Cases</Link> &gt;{" "}
          <span className="font-semibold">{category.category}</span>
        </nav>

        {/* Category Title */}
        <h1 className="text-3xl font-bold mb-6">{category.category}</h1>

        {/* Use Cases List (Styled Rows) */}
        <div className="flex flex-col gap-y-4">
          {category.usecases.map((uc) => (
            <Link
              key={uc.name}
              href={`/${locale}/usecases/${categorySlug}/${slugify(uc.name)}`}
            >
              <div className="flex items-center gap-4 p-4 bg-white dark:bg-gray-900 rounded-xl shadow hover:shadow-lg transition cursor-pointer">
                {/* Image placeholder */}
                <div className="w-32 h-32 bg-gray-200 dark:bg-gray-700 rounded-lg flex items-center justify-center">
                  <span className="text-gray-500 text-xs">Img</span>
                </div>

                {/* Text content */}
                <div className="flex-1">
                  <h2 className="text-lg font-semibold text-green-600">
                    {uc.name}
                  </h2>
                </div>
              </div>
            </Link>
          ))}
        </div>
      </main>

      <Footer />
    </div>
  );
}
