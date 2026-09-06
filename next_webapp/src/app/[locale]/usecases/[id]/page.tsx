"use client";

import React, { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { Link } from "@/i18n-navigation";
import { ArrowLeft, Download } from "lucide-react";
import Image from "next/image";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import NotebookRenderer from "@/components/NotebookRenderer";
import { apiFetch } from "@/lib/apiFetch";
import ErrorBoundary from "@/components/ErrorBoundary";

const UseCasePage: React.FC = () => {
  const params = useParams();
  const id = params?.id;

  const [useCase, setUseCase] = useState<any>(null);
  const [tags, setTags] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [notFound, setNotFound] = useState(false);

  // Notebook/HTML body — fetched separately from GridFS via the dedicated
  // content route once metadata confirms content_file_id is actually set,
  // never inlined in the GET /api/usecases/[id] envelope.
  const [notebookText, setNotebookText] = useState<string | null>(null);
  const [contentLoading, setContentLoading] = useState(false);

  useEffect(() => {
    if (!id) return;

    apiFetch<{ success: boolean; data: any }>(`/api/usecases/${id}`)
      .then((ucJson) => {
        if (!ucJson.success) {
          setNotFound(true);
          return;
        }

        setUseCase(ucJson.data);
        // Tags are embedded directly on the Mongo doc now (denormalized),
        // so there's no separate join/lookup call needed here any more.
        setTags((ucJson.data.tags ?? []).map((t: any) => t.name));
      })
      .catch(() => setNotFound(true)) // apiFetch already showed a toast
      .finally(() => setLoading(false));
  }, [id]);

  useEffect(() => {
    if (!id || !useCase?.content_file_id) return;

    setContentLoading(true);
    // This endpoint returns the raw notebook text, not the app's usual
    // {success, data} JSON envelope, so use apiFetch's text mode.
    apiFetch<string>(`/api/usecases/${id}/content`, { responseType: "text" })
      .then(setNotebookText)
      .catch(() => setNotebookText(null)) // apiFetch already showed a toast
      .finally(() => setContentLoading(false));
  }, [id, useCase?.content_file_id]);

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

          {useCase.created_by_name && (
            <p className="mb-6 text-sm text-gray-500 dark:text-gray-400">
              Created by{" "}
              <span className="font-medium text-gray-700 dark:text-gray-200">
                {useCase.created_by_name}
              </span>
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
            <div className="relative mb-8 aspect-[16/9] w-full overflow-hidden rounded-2xl shadow-md">
              <Image
                src={useCase.cover_img}
                alt={useCase.title}
                fill
                sizes="(max-width: 768px) 100vw, 1024px"
                className="object-cover"
              />
            </div>
          )}

          {useCase.content_file_id ? (
            contentLoading ? (
              <div className="mb-8 flex items-center justify-center rounded-2xl border border-gray-200 bg-gray-50 p-10 dark:border-gray-700 dark:bg-gray-900">
                <div className="h-6 w-6 animate-spin rounded-full border-4 border-green-500 border-t-transparent" />
              </div>
            ) : notebookText !== null ? (
              // content_type drives rendering directly now — no more
              // JSON.parse-and-see-if-it-throws sniffing.
              useCase.content_type === "html" ? (
                // legacy HTML fallback, unchanged from before — the string is
                // fetched from GridFS now instead of arriving inline, but the
                // srcDoc rendering mechanism (and its opaque-origin semantics)
                // is left exactly as-is.
                <iframe
                  srcDoc={notebookText}
                  className="mb-8 w-full rounded-2xl border border-gray-200 dark:border-gray-700"
                  style={{ height: "80vh", minHeight: "400px" }}
                  title={useCase.title}
                />
              ) : (
                <div className="mb-8 rounded-2xl border border-gray-200 bg-white p-6 text-gray-800 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-200 [&_*]:dark:text-gray-200">
                  <ErrorBoundary name="NotebookRenderer">
                    <NotebookRenderer content={notebookText} />
                  </ErrorBoundary>
                </div>
              )
            ) : (
              <div className="mb-8 rounded-2xl border border-gray-200 bg-gray-50 p-6 text-gray-600 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-300">
                Failed to load notebook content.
              </div>
            )
          ) : (
            <div className="mb-8 rounded-2xl border border-gray-200 bg-gray-50 p-6 text-gray-600 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-300">
              No notebook content available for this use case.
            </div>
          )}

          <div className="flex flex-wrap gap-3">
            <Link
              href="/usecases"
              className="inline-flex items-center gap-2 rounded-xl border border-gray-200 bg-white px-4 py-2 text-sm font-medium text-green-600 shadow-sm transition hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-700 dark:hover:bg-gray-600"
            >
              <ArrowLeft size={16} />
              Back to use cases
            </Link>

            {useCase.content_file_id && (
              <a
                href={`/api/usecases/${id}/content`}
                download={`${useCase.title}.ipynb`}
                className="inline-flex items-center gap-2 rounded-xl border border-gray-200 bg-white px-4 py-2 text-sm font-medium text-green-600 shadow-sm transition hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-700 dark:hover:bg-gray-600"
              >
                <Download size={16} />
                Download .ipynb
              </a>
            )}
          </div>
        </div>
      </main>
      <Footer />
    </>
  );
};

export default UseCasePage;
