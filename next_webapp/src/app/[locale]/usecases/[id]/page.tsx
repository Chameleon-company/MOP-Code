"use client";

import React, { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import { demoCaseStudies } from "../database";

const UseCasePage: React.FC = () => {
  const params = useParams();
  const id = params.id;

  const [htmlContent, setHtmlContent] = useState("");
  const [loading, setLoading] = useState(true);

  const useCase = demoCaseStudies.find(
    (item) => String(item.id) === String(id)
  );

  useEffect(() => {
    const loadHtml = async () => {
      if (!useCase?.htmlFile) {
        setLoading(false);
        return;
      }

      try {
        const response = await fetch(useCase.htmlFile);
        const html = await response.text();
        setHtmlContent(html);
      } catch (error) {
        console.error("Failed to load HTML:", error);
        setHtmlContent("<p>Failed to load content.</p>");
      } finally {
        setLoading(false);
      }
    };

    loadHtml();
  }, [useCase]);

  if (!useCase) {
    return (
      <>
        <Header />
        <main className="min-h-screen bg-[#f7f9fb] px-6 py-20 dark:bg-gray-900">
          <div className="mx-auto max-w-4xl rounded-3xl bg-white p-10 shadow-sm dark:bg-gray-800">
            <h1 className="text-3xl font-bold text-black dark:text-white">
              Use case not found
            </h1>
          </div>
        </main>
        <Footer />
      </>
    );
  }

  return (
    <>
      <Header />
      <main className="min-h-screen bg-[#f7f9fb] px-6 py-12 dark:bg-gray-900">
        <div className="mx-auto max-w-5xl rounded-[28px] border border-gray-200 bg-white p-8 shadow-sm dark:border-gray-700 dark:bg-gray-800 sm:p-10">
          <div className="mb-6 inline-flex items-center rounded-full bg-green-50 px-4 py-1.5 text-sm font-semibold text-green-700 dark:bg-green-900/30 dark:text-green-300">
            Open Data Use Case
          </div>

          <h1 className="mb-4 text-4xl font-bold tracking-tight text-black dark:text-white sm:text-5xl">
            {useCase.title}
          </h1>

          <p className="mb-6 text-lg text-gray-600 dark:text-gray-300">
            {useCase.description}
          </p>

          <div className="mb-8 flex flex-wrap gap-2">
            {useCase.tags?.map((tag, index) => (
              <span
                key={index}
                className="rounded-full bg-gray-100 px-3 py-1 text-sm text-gray-700 dark:bg-gray-700 dark:text-gray-200"
              >
                {tag}
              </span>
            ))}
          </div>

          {loading ? (
            <p className="text-lg text-gray-600 dark:text-gray-300">
              Loading content...
            </p>
          ) : (
            <div
              className="prose prose-lg max-w-none dark:prose-invert
                         prose-headings:font-bold
                         prose-img:rounded-2xl
                         prose-img:shadow-md"
              dangerouslySetInnerHTML={{ __html: htmlContent }}
            />
          )}
        </div>
      </main>
      <Footer />
    </>
  );
};

export default UseCasePage;