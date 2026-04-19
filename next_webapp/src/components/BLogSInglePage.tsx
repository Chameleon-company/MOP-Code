"use client";
import React from "react";
import { useParams } from "next/navigation";
import { ArrowLeft } from "lucide-react";
import { Link } from "@/i18n-navigation";
import { blogs, type BlogPost } from "@/utils/data";


function pickRelatedPosts(
  all: BlogPost[],
  currentId: number,
  count: number
): BlogPost[] {
  const others = all.filter((b) => b.id !== currentId);
  const n = others.length;
  if (n <= count) return others;
  const start = (currentId * 2) % n;
  return Array.from({ length: count }, (_, i) => others[(start + i) % n]);
}

function splitIntoNGroups<T>(items: T[], n: number): T[][] {
  const groups: T[][] = Array.from({ length: n }, () => []);
  if (items.length === 0 || n <= 0) return groups;
  const base = Math.floor(items.length / n);
  let rem = items.length % n;
  let idx = 0;
  for (let g = 0; g < n; g++) {
    const size = base + (rem > 0 ? 1 : 0);
    if (rem > 0) rem--;
    groups[g] = items.slice(idx, idx + size);
    idx += size;
  }
  return groups;
}

function BlogFigure({
  src,
  alt,
  priority,
  variant = "default",
  density = "compact",
}: {
  src: string;
  alt: string;
  priority?: boolean;
  variant?: "default" | "hero";
  density?: "comfortable" | "compact";
}) {
  const figureClass =
    variant === "hero"
      ? "mt-2 mb-10 w-full sm:mb-12 md:mb-14 sm:mt-3"
      : "my-12 w-full sm:my-14 md:my-16 lg:my-20";
  const sizeByVariant =
    density === "comfortable"
      ? variant === "hero"
        ? "max-w-[min(100%,44rem)] max-h-[min(64vh,680px)]"
        : "max-w-[min(100%,44rem)] max-h-[min(60vh,630px)]"
      : variant === "hero"
        ? "max-w-[min(100%,36rem)] max-h-[min(50vh,530px)]"
        : "max-w-[min(100%,34rem)] max-h-[min(46vh,480px)]";
  const imgClass =
    "mx-auto block h-auto w-auto rounded-2xl shadow-[0_16px_40px_-12px_rgba(0,0,0,0.15)] ring-1 ring-black/5 dark:shadow-[0_16px_40px_-12px_rgba(0,0,0,0.4)] dark:ring-white/10 md:rounded-3xl " +
    sizeByVariant;
  return (
    <figure className={figureClass}>
      <img
        src={src}
        alt={alt}
        loading={priority ? "eager" : "lazy"}
        decoding="async"
        className={imgClass}
      />
    </figure>
  );
}

function TextBlock({ paragraphs }: { paragraphs: string[] }) {
  if (paragraphs.length === 0) return null;
  return (
    <div className="space-y-6 text-[15px] leading-[1.75] text-gray-700 dark:text-gray-300 sm:text-base sm:leading-8">
      {paragraphs.map((p, i) => (
        <p key={i}>{p}</p>
      ))}
    </div>
  );
}

const BlogSinglePage: React.FC = () => {
  const { id } = useParams();
  const blog = blogs.find((b) => b.id.toString() === id);

  if (!blog) {
    return (
      <div className="flex flex-col items-center justify-center gap-4 min-h-screen px-6 text-gray-600 dark:text-gray-300">
        <p>Blog post not found :)</p>
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

  const paragraphs = blog.content
    .split(/\n\n+/)
    .map((p) => p.trim())
    .filter(Boolean);

  const [, ...inlineImages] = blog.images;
  const textGroups = splitIntoNGroups(paragraphs, inlineImages.length + 1);

  const relatedPosts = pickRelatedPosts(blogs, blog.id, 3);

  return (
    <article className="min-h-screen bg-white text-black dark:bg-[#141414] dark:text-white">
      <div className="mx-auto w-full max-w-7xl px-5 pb-16 pt-8 sm:px-8 sm:pb-20 sm:pt-10 md:px-12 md:pb-24 lg:px-16 xl:px-20">
        <header className="mb-8 text-left sm:mb-10 lg:mb-12">
          <p className="mb-5 text-[0.7rem] font-semibold uppercase tracking-[0.28em] text-[#2DBE6C] sm:text-xs sm:tracking-[0.35em]">
            Melbourne Open Playground · Insights
          </p>
          <h1 className="max-w-6xl text-pretty text-[1.5rem] font-bold leading-snug tracking-tight text-gray-950 dark:text-white sm:text-[1.875rem] md:text-[2.125rem] lg:text-[2.375rem]">
            {blog.title}
          </h1>
          <p className="mt-7 max-w-5xl text-base leading-[1.7] text-gray-600 dark:text-gray-400 md:mt-9 md:text-lg md:leading-relaxed lg:text-xl">
            {blog.description}
          </p>
          <div className="mt-10 flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-gray-200 pt-8 text-sm text-gray-500 dark:border-gray-700 dark:text-gray-400 md:mt-12 md:pt-10">
            <time className="font-medium text-gray-700 dark:text-gray-300" dateTime={blog.date}>
              {blog.date}
            </time>
            <span className="hidden h-4 w-px bg-gray-300 sm:block dark:bg-gray-600" aria-hidden />
            <span>By {blog.author}</span>
          </div>
        </header>

        <div className="mx-auto w-full max-w-5xl lg:max-w-6xl">
          <BlogFigure
            src={blog.image}
            alt={blog.title}
            priority
            variant="hero"
            density={blog.id === 3 ? "comfortable" : "compact"}
          />

          <TextBlock paragraphs={textGroups[0]} />

          <BlogFigure
            src={inlineImages[0]}
            alt={`${blog.title} — figure 2`}
            density="compact"
          />

          <TextBlock paragraphs={textGroups[1]} />

          <BlogFigure
            src={inlineImages[1]}
            alt={`${blog.title} — figure 3`}
            density="compact"
          />

          <TextBlock paragraphs={textGroups[2]} />
        </div>

        <div className="mx-auto mt-14 flex w-full max-w-5xl justify-start sm:mt-16 lg:max-w-6xl">
          <Link
            href="/blog"
            className="inline-flex items-center gap-2 rounded-xl border border-gray-200 bg-white px-4 py-2 text-sm font-medium text-[#2DBE6C] shadow-sm transition-colors hover:bg-gray-50 dark:border-gray-600 dark:bg-[#1f1f1f] dark:hover:bg-[#2a2a2a]"
          >
            <ArrowLeft className="h-4 w-4 shrink-0" aria-hidden />
            Back to blogs
          </Link>
        </div>

        {relatedPosts.length > 0 ? (
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
              {relatedPosts.map((b) => (
                <li key={b.id} className="flex">
                  <Link
                    href={`/blog/${b.id}`}
                    className="group flex w-full flex-col overflow-hidden rounded-xl bg-white shadow-[0_4px_24px_-4px_rgba(0,0,0,0.08)] ring-1 ring-black/[0.06] transition hover:shadow-[0_12px_40px_-8px_rgba(0,0,0,0.14)] dark:bg-[#1f1f1f] dark:ring-white/[0.08] dark:hover:shadow-[0_12px_40px_-8px_rgba(0,0,0,0.45)]"
                  >
                    <div className="relative aspect-[16/10] w-full shrink-0 overflow-hidden bg-gray-100 dark:bg-gray-800">
                      <img
                        src={b.image}
                        alt={b.title}
                        className="h-full w-full object-cover transition duration-300 group-hover:scale-[1.02]"
                        loading="lazy"
                        decoding="async"
                      />
                      {b.category ? (
                        <span className="absolute bottom-3 left-3 rounded-full bg-[#1a1a1a]/90 px-3 py-1.5 text-[13px] font-medium leading-tight text-white shadow-sm backdrop-blur-[2px] dark:bg-black/80">
                          {b.category}
                        </span>
                      ) : null}
                    </div>
                    <div className="flex flex-1 flex-col p-5 sm:p-6">
                      <h3 className="line-clamp-3 text-left text-base font-bold leading-snug text-gray-900 dark:text-white">
                        {b.title}
                      </h3>
                      <p className="mt-3 line-clamp-3 text-left text-sm leading-relaxed text-gray-600 dark:text-gray-400">
                        {b.description}
                      </p>
                    </div>
                  </Link>
                </li>
              ))}
            </ul>
          </section>
        ) : null}
      </div>
    </article>
  );
};

export default BlogSinglePage;
