"use client";

import React, { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import Header from "@/components/Header";
import Footer from "@/components/Footer";

const UseCasePage: React.FC = () => {
  const params = useParams();
  const id = params?.id;

  const [useCase, setUseCase] = useState<any>(null);
  const [tags, setTags] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [notFound, setNotFound] = useState(false);

  useEffect(() => {
    if (!id) return;

    Promise.all([
      fetch(`/api/usecases/${id}`).then((r) => r.json()),
      fetch(`/api/usecases/${id}/tags`).then((r) => r.json()),
    ])
      .then(([ucJson, tagsJson]) => {
        if (!ucJson.success) {
          setNotFound(true);
          return;
        }
        setUseCase(ucJson.data);
        if (tagsJson.success && Array.isArray(tagsJson.data)) {
          setTags(tagsJson.data.map((t: any) => t.name));
        }
      })
      .catch(() => setNotFound(true))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) {
    return (
      <>
        <Header />
        <main className="flex min-h-screen items-center justify-center bg-[#f7f9fb] dark:bg-gray-900">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-green-500 border-t-transparent" />
        </main>
        <Footer />
      </>
    );
  }

  if (notFound || !useCase) {
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

          {useCase.description && (
            <p className="mb-6 text-lg text-gray-600 dark:text-gray-300">
              {useCase.description}
            </p>
          )}

          {tags.length > 0 && (
            <div className="mb-8 flex flex-wrap gap-2">
              {tags.map((tag) => (
                <span
                  key={tag}
                  className="rounded-full bg-gray-100 px-3 py-1 text-sm text-gray-700 dark:bg-gray-700 dark:text-gray-200"
                >
                  {tag}
                </span>
              ))}
            </div>
          )}

          {useCase.cover_img && (
            <img
              src={useCase.cover_img}
              alt={useCase.title}
              className="mb-8 w-full rounded-2xl object-cover shadow-md"
            />
          )}

          <Link
            href="/usecases"
            className="inline-flex items-center gap-2 rounded-xl border border-gray-200 bg-white px-4 py-2 text-sm font-medium text-green-600 shadow-sm transition hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-700 dark:hover:bg-gray-600"
          >
            <ArrowLeft size={16} />
            Back to use cases
          </Link>
        </div>
      </main>
      <Footer />
    </>
  );
};

export default UseCasePage;
