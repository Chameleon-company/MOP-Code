// "use client";

// import { useState, useEffect, useCallback } from "react";
// import Header from "@/components/Header";
// import Footer from "@/components/Footer";
// import BlogCard from "@/components/BlogCard";
// import { ChevronLeft, ChevronRight } from "lucide-react";

// const PAGE_SIZE = 9;

// interface Blog {
//   id: number;
//   title: string;
//   description: string | null;
//   cover_img: string | null;
//   published_date: string | null;
// }

// export default function BlogListingPage() {
//   const [blogs, setBlogs] = useState<Blog[]>([]);
//   const [loading, setLoading] = useState(true);
//   const [page, setPage] = useState(1);
//   const [totalPages, setTotalPages] = useState(1);
//   const [total, setTotal] = useState(0);

//   const fetchBlogs = useCallback(async (currentPage: number) => {
//     setLoading(true);
//     try {
//       const params = new URLSearchParams({
//         page: String(currentPage),
//         pageSize: String(PAGE_SIZE),
//       });
//       const res = await fetch(`/api/home/blogs?${params}`);
//       const json = await res.json();
//       if (json.success) {
//         setBlogs(json.data ?? []);
//         setTotal(json.pagination?.total ?? 0);
//         setTotalPages(json.pagination?.totalPages ?? 1);
//       }
//     } catch (e) {
//       console.error(e);
//     } finally {
//       setLoading(false);
//     }
//   }, []);

//   useEffect(() => {
//     fetchBlogs(page);
//   }, [page, fetchBlogs]);

//   return (
//     <div className="flex min-h-screen flex-col">
//       <Header />

//       <main className="flex-1 bg-white dark:bg-[#1C1C1C]">
//         <section className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
//           {/* Heading */}
//           <div className="mb-12 text-center">
//             <h1 className="mb-3 text-4xl font-bold text-gray-900 dark:text-white md:text-5xl">
//               Blog
//             </h1>
//             <p className="text-sm text-gray-500 dark:text-gray-400 md:text-base">
//               Insights, updates, and expert tips
//             </p>
//           </div>

//           {/* Responsive card grid: 1 → 2 → 3 columns */}
//           {blogs.length > 0 ? (
//             <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
//               {blogs.map((blog) => (
//                 <BlogCard
//                   key={blog.id}
//                   id={String(blog.id)}
//                   title={blog.title ?? ""}
//                   description={blog.description ?? ""}
//                   image={blog.cover_img ?? ""}
//                   category={blog.published_date ?? ""}
//                 />
//               ))}
//             </div>
//           ) : (
//             <div className="flex min-h-[250px] items-center justify-center rounded-2xl border border-dashed border-gray-300 bg-gray-50 px-6 py-12 text-center dark:border-gray-700 dark:bg-[#242424]">
//               <p className="text-base font-medium text-gray-500 dark:text-gray-400">
//                 No data available at the moment
//               </p>
//             </div>
//           )}
//         </section>
//       </main>

//       <Footer />
//     </div>
//   );
// }
"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import BlogCard from "@/components/BlogCard";
import { ChevronLeft, ChevronRight, Search } from "lucide-react";

const PAGE_SIZE = 9;
type SearchMode = "title" | "content";

interface Blog {
  id: number;
  title: string;
  description: string | null;
  cover_img: string | null;
  published_date: string | null;
}

export default function BlogListingPage() {
  const [blogs, setBlogs] = useState<Blog[]>([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  const [searchTerm, setSearchTerm] = useState("");
  const [searchMode, setSearchMode] = useState<SearchMode>("title");
  const [activeSearch, setActiveSearch] = useState<{ term: string; mode: SearchMode }>({ term: "", mode: "title" });

  const fetchBlogs = useCallback(async (currentPage: number, term: string, mode: SearchMode) => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        page: String(currentPage),
        pageSize: String(PAGE_SIZE),
      });
      if (term) {
        params.set("search", term);
        params.set("search_by", mode);
      }
      const res = await fetch(`/api/home/blogs?${params}`);
      const json = await res.json();
      if (json.success) {
        setBlogs(json.data ?? []);
        setTotal(json.pagination?.total ?? 0);
        setTotalPages(json.pagination?.totalPages ?? 1);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchBlogs(page, activeSearch.term, activeSearch.mode);
  }, [page, activeSearch, fetchBlogs]);

  // Debounce: update activeSearch 350ms after typing stops (skip first render)
  const isFirstRender = useRef(true);
  useEffect(() => {
    if (isFirstRender.current) { isFirstRender.current = false; return; }
    const timer = setTimeout(() => {
      setPage(1);
      setActiveSearch({ term: searchTerm, mode: searchMode });
    }, 350);
    return () => clearTimeout(timer);
  }, [searchTerm, searchMode]);

  // Submit: bypass debounce, fire immediately
  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    setActiveSearch({ term: searchTerm, mode: searchMode });
  };

  const handleReset = () => {
    setSearchTerm("");
    setSearchMode("title");
    setPage(1);
    setActiveSearch({ term: "", mode: "title" });
  };

  return (
    <div className="flex min-h-screen flex-col">
      <Header />

      <main className="flex-1 bg-white dark:bg-[#1C1C1C]">
        <section className="relative overflow-hidden bg-gradient-to-br from-green-700 via-green-600 to-green-500 px-4 pb-20 pt-20 text-center dark:from-green-900 dark:via-green-800 dark:to-green-700">
          <div className="pointer-events-none absolute -left-16 -top-16 h-64 w-64 rounded-full bg-white/10 blur-2xl" />
          <div className="pointer-events-none absolute -bottom-12 -right-12 h-80 w-80 rounded-full bg-white/10 blur-2xl" />

          <div className="relative mx-auto max-w-3xl">
            <span className="mb-4 inline-block rounded-full bg-white/20 px-4 py-1 text-xs font-semibold uppercase tracking-widest text-green-100">
              Melbourne Open Data
            </span>

            <h1
              className="mb-5 text-5xl text-white drop-shadow-sm sm:text-6xl"
              style={{
                fontWeight: 900,
                fontStyle: "normal",
                fontFamily: "'Barlow Condensed', sans-serif",
                letterSpacing: "-0.02em",
                textShadow: "2px 2px 8px rgba(0,0,0,0.35)",
              }}
            >
              Blog
            </h1>

            <p className="mx-auto max-w-2xl text-lg leading-relaxed text-green-100">
              Insights, updates, and expert tips
            </p>
          </div>

          <div className="absolute bottom-0 left-0 right-0 overflow-hidden leading-none">
            <svg
              viewBox="0 0 1440 40"
              xmlns="http://www.w3.org/2000/svg"
              className="block w-full fill-white dark:fill-[#1C1C1C]"
            >
              <path d="M0,20 C360,40 1080,0 1440,20 L1440,40 L0,40 Z" />
            </svg>
          </div>
        </section>

        <section className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
          {/* Search bar */}
          <div className="mb-8 rounded-[24px] border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-700 dark:bg-[#242424] sm:p-5">
            <form onSubmit={handleSearch} className="flex flex-col gap-4 lg:flex-row lg:items-center">
              <div className="relative flex-1">
                <Search size={20} className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400" />
                <input
                  type="search"
                  placeholder="Search blogs..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="h-14 w-full rounded-2xl border border-gray-200 bg-gray-50 pl-12 pr-4 text-sm outline-none transition focus:border-green-500 focus:bg-white dark:border-gray-600 dark:bg-[#1C1C1C] dark:text-white dark:focus:bg-[#1C1C1C]"
                />
              </div>
              <div className="flex flex-col gap-3 sm:flex-row">
                <select
                  value={searchMode}
                  onChange={(e) => setSearchMode(e.target.value as SearchMode)}
                  className="h-14 min-w-[200px] rounded-2xl border border-gray-200 bg-white px-4 text-sm outline-none transition focus:border-green-500 dark:border-gray-600 dark:bg-[#242424] dark:text-white"
                >
                  <option value="title">Search by title</option>
                  <option value="content">Search by content</option>
                </select>
                <button
                  type="submit"
                  className="h-14 rounded-2xl bg-green-600 px-6 text-sm font-semibold text-white transition hover:bg-green-700"
                >
                  Search
                </button>
                <button
                  type="button"
                  onClick={handleReset}
                  className="h-14 rounded-2xl border border-gray-200 bg-white px-6 text-sm font-semibold text-gray-700 transition hover:bg-gray-50 dark:border-gray-600 dark:bg-[#242424] dark:text-white dark:hover:bg-[#2e2e2e]"
                >
                  Reset
                </button>
              </div>
            </form>
          </div>

          {activeSearch.term && !loading && (
            <p className="mb-6 text-sm text-gray-500 dark:text-gray-400">
              {total} result{total !== 1 ? "s" : ""} for &ldquo;{activeSearch.term}&rdquo;
            </p>
          )}

          {loading ? (
            <div className="flex justify-center py-20">
              <div className="h-10 w-10 animate-spin rounded-full border-4 border-green-200 border-t-green-600" />
            </div>
          ) : blogs.length > 0 ? (
            <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
              {blogs.map((blog) => (
                <BlogCard
                  key={blog.id}
                  id={String(blog.id)}
                  title={blog.title ?? ""}
                  description={blog.description ?? ""}
                  image={blog.cover_img ?? ""}
                  category={blog.published_date ?? ""}
                />
              ))}
            </div>
          ) : blogs.length > 0 ? (
            <>
              <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
                {blogs.map((blog) => (
                  <BlogCard
                    key={blog.id}
                    id={String(blog.id)}
                    title={blog.title ?? ""}
                    description={blog.description ?? ""}
                    image={blog.cover_img ?? ""}
                    category={blog.published_date ?? ""}
                  />
                ))}
              </div>

              {totalPages > 1 && (
                <div className="mt-12 flex items-center justify-center gap-3">
                  <button
                    onClick={() => setPage((prev) => Math.max(prev - 1, 1))}
                    disabled={page === 1}
                    className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-gray-200 bg-white text-gray-700 transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-gray-600 dark:bg-gray-800 dark:text-white dark:hover:bg-gray-700"
                  >
                    <ChevronLeft size={18} />
                  </button>

                  <span className="rounded-full bg-green-50 px-4 py-2 text-sm font-semibold text-green-700 dark:bg-green-900/30 dark:text-green-300">
                    Page {page} of {totalPages}
                  </span>

                  <button
                    onClick={() =>
                      setPage((prev) => Math.min(prev + 1, totalPages))
                    }
                    disabled={page === totalPages}
                    className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-gray-200 bg-white text-gray-700 transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-gray-600 dark:bg-gray-800 dark:text-white dark:hover:bg-gray-700"
                  >
                    <ChevronRight size={18} />
                  </button>
                </div>
              )}
            </>
          ) : (
            <div className="flex min-h-[250px] items-center justify-center rounded-2xl border border-dashed border-gray-300 bg-gray-50 px-6 py-12 text-center dark:border-gray-700 dark:bg-[#242424]">
              <p className="text-base font-medium text-gray-500 dark:text-gray-400">
                {activeSearch.term ? `No blogs found for "${activeSearch.term}"` : "No data available at the moment"}
              </p>
            </div>
          )}
        </section>
      </main>

      <Footer />
    </div>
  );
}