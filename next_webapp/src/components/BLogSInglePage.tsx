"use client";

import React, { useEffect, useState } from "react";
import { ArrowLeft } from "lucide-react";
import { Link } from "@/i18n-navigation";

interface Blog {
  id: number;
  title: string;
  description: string | null;
  cover_img: string | null;
  published_date: string | null;
  content: string | null;
  author: string;
}

interface RelatedBlog {
  id: number;
  title: string;
  description: string | null;
  cover_img: string | null;
  published_date: string | null;
}

function formatDate(dateStr: string | null): string {
  if (!dateStr) return "";
  const d = new Date(dateStr);
  return d.toLocaleDateString("en-AU", { year: "numeric", month: "long", day: "numeric" });
}

const BlogSinglePage: React.FC<{ id: string }> = ({ id }) => {

  const [blog, setBlog] = useState<Blog | null>(null);
  const [related, setRelated] = useState<RelatedBlog[]>([]);
  const [loading, setLoading] = useState(true);
  const [notFound, setNotFound] = useState(false);

  useEffect(() => {
    if (!id) return;

    const fetchBlog = async () => {
      setLoading(true);
      try {
        const res = await fetch(`/api/home/blogs/${id}`);
        const json = await res.json();

        if (!res.ok || !json.success) {
          setNotFound(true);
          return;
        }

        setBlog(json.data);

        // Random recommendations (server pools up to 800, shuffles, returns 3)
        const relRes = await fetch(
          `/api/home/blogs?recommend=1&excludeId=${encodeURIComponent(String(json.data.id))}&take=3`
        );
        const relJson = await relRes.json();
        if (relJson.success && Array.isArray(relJson.data)) {
          setRelated(relJson.data as RelatedBlog[]);
        }
      } catch (e) {
        console.error(e);
        setNotFound(true);
      } finally {
        setLoading(false);
      }
    };

    fetchBlog();
  }, [id]);

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-green-500 border-t-transparent" />
      </div>
    );
  }

  if (notFound || !blog) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-4 px-6 text-gray-600 dark:text-gray-300">
        <p>Blog post not found.</p>
        <Link
          href="/blog"
          className="inline-flex items-center gap-2 rounded-xl bg-green-500 px-4 py-2 text-sm font-medium text-white hover:bg-green-600 transition-colors"
        >
          <ArrowLeft className="h-4 w-4" aria-hidden />
          Back to blogs
        </Link>
      </div>
    );
  }

  return (
    <article className="min-h-screen bg-white text-black dark:bg-[#141414] dark:text-white">
      <div className="mx-auto w-full max-w-7xl px-5 pb-16 pt-8 sm:px-8 sm:pb-20 sm:pt-10 md:px-12 md:pb-24 lg:px-16 xl:px-20">

        {/* ── Header ── */}
        <header className="mb-8 text-left sm:mb-10 lg:mb-12">
          <p className="mb-5 text-[0.7rem] font-semibold uppercase tracking-[0.28em] text-[#2DBE6C] sm:text-xs sm:tracking-[0.35em]">
            Melbourne Open Playground · Insights
          </p>

          <h1 className="max-w-6xl text-pretty text-[1.5rem] font-bold leading-snug tracking-tight text-gray-950 dark:text-white sm:text-[1.875rem] md:text-[2.125rem] lg:text-[2.375rem]">
            {blog.title}
          </h1>

          {blog.description && (
            <p className="mt-7 max-w-5xl text-base leading-[1.7] text-gray-600 dark:text-gray-400 md:mt-9 md:text-lg md:leading-relaxed lg:text-xl">
              {blog.description}
            </p>
          )}

          <div className="mt-10 flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-gray-200 pt-8 text-sm text-gray-500 dark:border-gray-700 dark:text-gray-400 md:mt-12 md:pt-10">
            {blog.published_date && (
              <>
                <time className="font-medium text-gray-700 dark:text-gray-300" dateTime={blog.published_date}>
                  {formatDate(blog.published_date)}
                </time>
                <span className="hidden h-4 w-px bg-gray-300 sm:block dark:bg-gray-600" aria-hidden />
              </>
            )}
            <span>By {blog.author}</span>
          </div>
        </header>

        {/* ── Cover image ── */}
        {blog.cover_img && (
          <figure className="mt-2 mb-10 w-full sm:mb-12 md:mb-14">
            <img
              src={blog.cover_img}
              alt={blog.title}
              loading="eager"
              decoding="async"
              className="mx-auto block h-auto w-full max-h-[520px] rounded-2xl object-cover shadow-[0_16px_40px_-12px_rgba(0,0,0,0.15)] ring-1 ring-black/5 dark:shadow-[0_16px_40px_-12px_rgba(0,0,0,0.4)] dark:ring-white/10 md:rounded-3xl"
            />
          </figure>
        )}

        {/* ── CKEditor HTML content ── */}
        {blog.content && (
          <div
            className="mx-auto w-full max-w-5xl lg:max-w-6xl prose-blog"
            dangerouslySetInnerHTML={{ __html: blog.content }}
          />
        )}

        {/* ── Back link ── */}
        <div className="mx-auto mt-14 flex w-full max-w-5xl justify-start sm:mt-16 lg:max-w-6xl">
          <Link
            href="/blog"
            className="inline-flex items-center gap-2 rounded-xl border border-gray-200 bg-white px-4 py-2 text-sm font-medium text-[#2DBE6C] shadow-sm transition-colors hover:bg-gray-50 dark:border-gray-600 dark:bg-[#1f1f1f] dark:hover:bg-[#2a2a2a]"
          >
            <ArrowLeft className="h-4 w-4 shrink-0" aria-hidden />
            Back to blogs
          </Link>
        </div>

        {/* ── Continue exploring ── */}
        {related.length > 0 && (
          <section
            className="mx-auto mt-12 w-full max-w-5xl border-t border-gray-200 pt-10 dark:border-gray-700 sm:mt-14 sm:pt-12 lg:max-w-6xl"
            aria-labelledby="continue-reading-heading"
          >
            <div className="mb-8 text-center sm:mb-10">
              <h2
                id="continue-reading-heading"
                className="text-2xl font-bold tracking-tight text-gray-950 dark:text-white sm:text-3xl"
              >
                Continue exploring
              </h2>
              <p className="mx-auto mt-2 max-w-xl text-sm leading-relaxed text-gray-600 dark:text-gray-400 sm:text-base">
                Insights, updates, and expert tips from the Melbourne Open Playground
              </p>
            </div>

            <ul className="grid grid-cols-1 gap-6 md:grid-cols-3 md:gap-5 lg:gap-6">
              {related.map((b) => (
                <li key={b.id} className="flex">
                  <Link
                    href={`/blog/${b.id}`}
                    className="group flex w-full flex-col overflow-hidden rounded-xl bg-white shadow-[0_4px_24px_-4px_rgba(0,0,0,0.08)] ring-1 ring-black/[0.06] transition hover:shadow-[0_12px_40px_-8px_rgba(0,0,0,0.14)] dark:bg-[#1f1f1f] dark:ring-white/[0.08]"
                  >
                    <div className="relative aspect-[16/10] w-full shrink-0 overflow-hidden bg-gray-100 dark:bg-gray-800">
                      {b.cover_img ? (
                        <img
                          src={b.cover_img}
                          alt={b.title}
                          className="h-full w-full object-cover transition duration-300 group-hover:scale-[1.02]"
                          loading="lazy"
                          decoding="async"
                        />
                      ) : (
                        <div className="h-full w-full bg-gray-200 dark:bg-gray-700" />
                      )}
                      {b.published_date && (
                        <span className="absolute bottom-3 left-3 rounded-full bg-[#1a1a1a]/90 px-3 py-1.5 text-[13px] font-medium text-white backdrop-blur-[2px]">
                          {formatDate(b.published_date)}
                        </span>
                      )}
                    </div>
                    <div className="flex flex-1 flex-col p-5 sm:p-6">
                      <h3 className="line-clamp-3 text-left text-base font-bold leading-snug text-gray-900 dark:text-white">
                        {b.title}
                      </h3>
                      {b.description && (
                        <p className="mt-3 line-clamp-3 text-left text-sm leading-relaxed text-gray-600 dark:text-gray-400">
                          {b.description}
                        </p>
                      )}
                    </div>
                  </Link>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </article>
  );
};

export default BlogSinglePage;
