
// "use client";

// import React, { useEffect, useState } from "react";
// import { useParams } from "next/navigation";
// import Link from "next/link";
// import { ArrowLeft } from "lucide-react";
// import Header from "@/components/Header";
// import Footer from "@/components/Footer";

// const CategoryUseCasesPage = () => {
//   const params = useParams();

//   const locale = params?.locale as string;
//   const categoryId = params?.id as string;

//   const [category, setCategory] = useState<any>(null);
//   const [useCases, setUseCases] = useState<any[]>([]);
//   const [loading, setLoading] = useState(true);
//   const [error, setError] = useState("");

//   useEffect(() => {
//     if (!categoryId) return;

//     const fetchCategoryUseCases = async () => {
//       try {
//         setLoading(true);
//         setError("");

//         // Fetch categories from the same working API used on homepage
//         const categoryRes = await fetch("/api/home/categories");

//         if (!categoryRes.ok) {
//           throw new Error("Failed to fetch categories");
//         }

//         const categoryJson = await categoryRes.json();

//         const allCategories = categoryJson.success
//           ? categoryJson.data || []
//           : Array.isArray(categoryJson)
//           ? categoryJson
//           : [];

//         const selectedCategory = allCategories.find(
//           (cat: any) => String(cat.id) === String(categoryId)
//         );

//         setCategory(selectedCategory || null);

//         // Fetch use cases by selected category
//         const useCasesRes = await fetch(
//           `/api/usecases?category_id=${categoryId}`
//         );

//         if (!useCasesRes.ok) {
//           throw new Error("Failed to fetch category use cases");
//         }

//         const useCasesJson = await useCasesRes.json();

//         const categoryUseCases = useCasesJson.success
//           ? useCasesJson.data || []
//           : Array.isArray(useCasesJson)
//           ? useCasesJson
//           : useCasesJson.data || [];

//         setUseCases(categoryUseCases);
//       } catch (err) {
//         console.error("Category page error:", err);
//         setError("Unable to load use cases for this category.");
//       } finally {
//         setLoading(false);
//       }
//     };

//     fetchCategoryUseCases();
//   }, [categoryId]);

//   return (
//     <>
//       <Header />

//       <main className="min-h-screen bg-[#F6F9FC] dark:bg-[#263238] text-black dark:text-white">
//         {/* Hero Banner */}
//         <section className="relative overflow-hidden bg-gradient-to-r from-green-700 via-green-600 to-green-500 px-6 py-20 text-center text-white md:py-24">
//           <div className="mx-auto max-w-4xl">
//             <p className="mb-4 text-xs font-bold uppercase tracking-[0.25em] text-green-100">
//               Melbourne Open Data
//             </p>

//             <h1 className="text-4xl font-extrabold leading-tight drop-shadow-md md:text-6xl">
//               {category?.category_name || category?.name || "Use Cases"}
//             </h1>

//             {category?.description ? (
//               <p className="mx-auto mt-6 max-w-3xl text-base leading-8 text-green-50 md:text-lg">
//                 {category.description}
//               </p>
//             ) : (
//               <p className="mx-auto mt-6 max-w-3xl text-base leading-8 text-green-50 md:text-lg">
//                 Explore use cases connected to this category and discover how
//                 open data supports smarter city planning and decision-making.
//               </p>
//             )}

//           {!loading && !error && (
//   <div className="mt-7 flex justify-center">
//     <div className="inline-flex items-center gap-2 rounded-full bg-white px-4 py-2 text-sm font-semibold text-green-700 shadow-md">
//       <span className="flex h-6 w-6 items-center justify-center rounded-full bg-green-100 text-xs font-bold text-green-700">
//         {useCases.length}
//       </span>
//       <span>
//         {useCases.length === 1 ? "Use Case Found" : "Use Cases Found"}
//       </span>
//     </div>
//   </div>
// )}
//           </div>

//           {/* Bottom curve/wave effect */}
//           <div className="absolute bottom-0 left-0 h-10 w-full bg-[#F6F9FC] dark:bg-[#263238] [clip-path:polygon(0_70%,100%_25%,100%_100%,0_100%)]" />
//         </section>

//         {/* Content */}
//         <section className="px-6 py-12">
//           <div className="mx-auto max-w-7xl">
//             <Link
//               href={`/${locale}`}
//               className="mb-8 inline-flex items-center gap-2 rounded-full bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm transition hover:bg-gray-100 hover:text-black dark:bg-[#37474F] dark:text-gray-200 dark:hover:bg-[#455A64]"
//             >
//               <ArrowLeft size={18} />
//               Back to Home
//             </Link>

//             {loading && (
//               <div className="flex min-h-[300px] items-center justify-center rounded-3xl bg-white shadow-sm dark:bg-[#37474F]">
//                 <div className="text-center">
//                   <div className="mx-auto mb-4 h-9 w-9 animate-spin rounded-full border-4 border-green-500 border-t-transparent" />
//                   <p className="text-gray-600 dark:text-gray-300">
//                     Loading use cases...
//                   </p>
//                 </div>
//               </div>
//             )}

//             {!loading && error && (
//               <div className="rounded-2xl border border-red-100 bg-red-50 p-6 text-red-700 shadow-sm">
//                 {error}
//               </div>
//             )}

//             {!loading && !error && useCases.length === 0 && (
//               <div className="rounded-3xl bg-white p-12 text-center shadow-sm dark:bg-[#37474F]">
//                 <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
//                   No use cases found
//                 </h2>
//                 <p className="mt-2 text-gray-600 dark:text-gray-300">
//                   There are currently no use cases available in this category.
//                 </p>
//               </div>
//             )}

//             {!loading && !error && useCases.length > 0 && (
//               <div className="grid items-stretch gap-6 sm:grid-cols-2 lg:grid-cols-3">
//                 {useCases.map((item) => (
//                   <Link
//                     key={item.id}
//                     href={`/${locale}/usecases/${item.id}`}
//                     className="group flex h-full min-h-[390px] flex-col overflow-hidden rounded-3xl bg-white shadow-sm transition hover:-translate-y-1 hover:shadow-xl dark:bg-[#37474F]"
//                   >
//                     <div className="h-52 w-full overflow-hidden bg-gray-200 dark:bg-[#455A64]">
//                       <img
//                         src={
//                           item.cover_img ||
//                           item.cover_image ||
//                           item.image ||
//                           "/img/insights/eco.webp"
//                         }
//                         alt={item.title || item.name || "Use case"}
//                         className="h-full w-full object-cover transition duration-300 group-hover:scale-105"
//                       />
//                     </div>

//                     <div className="flex flex-1 flex-col p-6">
//                       <h2 className="mb-3 line-clamp-2 text-xl font-bold text-gray-900 dark:text-white">
//                         {item.title || item.name}
//                       </h2>

//                       <p className="line-clamp-3 text-sm leading-6 text-gray-600 dark:text-gray-300">
//                         {item.description || item.short_description || ""}
//                       </p>

//                       <div className="mt-auto pt-5 text-sm font-semibold text-black transition group-hover:text-green-600 dark:text-white dark:group-hover:text-green-400">
//                         View Use Case →
//                       </div>
//                     </div>
//                   </Link>
//                 ))}
//               </div>
//             )}
//           </div>
//         </section>
//       </main>

//       <Footer />
//     </>
//   );
// };

// export default CategoryUseCasesPage;
"use client";

import React, { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, ChevronLeft, ChevronRight } from "lucide-react";
import Header from "@/components/Header";
import Footer from "@/components/Footer";

const PAGE_SIZE = 9;

const CategoryUseCasesPage = () => {
  const params = useParams();

  const locale = params?.locale as string;
  const categoryId = params?.id as string;

  const [category, setCategory] = useState<any>(null);
  const [useCases, setUseCases] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [page, setPage] = useState(1);

  const totalPages = Math.ceil(useCases.length / PAGE_SIZE);

  const paginatedUseCases = useCases.slice(
    (page - 1) * PAGE_SIZE,
    page * PAGE_SIZE
  );

  useEffect(() => {
    if (!categoryId) return;

    const fetchCategoryUseCases = async () => {
      try {
        setLoading(true);
        setError("");
        setPage(1);

        // Fetch categories from the same working API used on homepage
        const categoryRes = await fetch("/api/home/categories");

        if (!categoryRes.ok) {
          throw new Error("Failed to fetch categories");
        }

        const categoryJson = await categoryRes.json();

        const allCategories = categoryJson.success
          ? categoryJson.data || []
          : Array.isArray(categoryJson)
          ? categoryJson
          : [];

        const selectedCategory = allCategories.find(
          (cat: any) => String(cat.id) === String(categoryId)
        );

        setCategory(selectedCategory || null);

        // Fetch use cases by selected category
        const useCasesRes = await fetch(
          `/api/usecases?category_id=${categoryId}`
        );

        if (!useCasesRes.ok) {
          throw new Error("Failed to fetch category use cases");
        }

        const useCasesJson = await useCasesRes.json();

        const categoryUseCases = useCasesJson.success
          ? useCasesJson.data || []
          : Array.isArray(useCasesJson)
          ? useCasesJson
          : useCasesJson.data || [];

        setUseCases(categoryUseCases);
      } catch (err) {
        console.error("Category page error:", err);
        setError("Unable to load use cases for this category.");
      } finally {
        setLoading(false);
      }
    };

    fetchCategoryUseCases();
  }, [categoryId]);

  return (
    <>
      <Header />

      <main className="min-h-screen bg-[#F6F9FC] text-black dark:bg-[#263238] dark:text-white">
        {/* Hero Banner */}
        <section className="relative overflow-hidden bg-gradient-to-br from-green-700 via-green-600 to-green-500 px-4 pb-20 pt-20 text-center dark:from-green-900 dark:via-green-800 dark:to-green-700">
          <div className="pointer-events-none absolute -left-16 -top-16 h-64 w-64 rounded-full bg-white/10 blur-2xl" />
          <div className="pointer-events-none absolute -bottom-12 -right-12 h-80 w-80 rounded-full bg-white/10 blur-2xl" />

          <div className="relative mx-auto max-w-4xl">
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
              {category?.category_name || category?.name || "Use Cases"}
            </h1>

            {category?.description ? (
              <p className="mx-auto max-w-3xl text-lg leading-relaxed text-green-100">
                {category.description}
              </p>
            ) : (
              <p className="mx-auto max-w-3xl text-lg leading-relaxed text-green-100">
                Explore use cases connected to this category and discover how
                open data supports smarter city planning and decision-making.
              </p>
            )}

            {!loading && !error && (
              <p className="mt-4 text-sm font-semibold text-green-100">
                {useCases.length}{" "}
                {useCases.length === 1 ? "use case found" : "use cases found"}
              </p>
            )}
          </div>

          <div className="absolute bottom-0 left-0 right-0 overflow-hidden leading-none">
            <svg
              viewBox="0 0 1440 40"
              xmlns="http://www.w3.org/2000/svg"
              className="block w-full fill-[#F6F9FC] dark:fill-[#263238]"
            >
              <path d="M0,20 C360,40 1080,0 1440,20 L1440,40 L0,40 Z" />
            </svg>
          </div>
        </section>

        {/* Content */}
        <section className="px-6 py-12">
          <div className="mx-auto max-w-7xl">
            <Link
              href={`/${locale}`}
              className="mb-8 inline-flex items-center gap-2 rounded-full bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm transition hover:bg-gray-100 hover:text-black dark:bg-[#37474F] dark:text-gray-200 dark:hover:bg-[#455A64]"
            >
              <ArrowLeft size={18} />
              Back to Home
            </Link>

            {loading && (
              <div className="flex min-h-[300px] items-center justify-center rounded-3xl bg-white shadow-sm dark:bg-[#37474F]">
                <div className="text-center">
                  <div className="mx-auto mb-4 h-9 w-9 animate-spin rounded-full border-4 border-green-500 border-t-transparent" />
                  <p className="text-gray-600 dark:text-gray-300">
                    Loading use cases...
                  </p>
                </div>
              </div>
            )}

            {!loading && error && (
              <div className="rounded-2xl border border-red-100 bg-red-50 p-6 text-red-700 shadow-sm">
                {error}
              </div>
            )}

            {!loading && !error && useCases.length === 0 && (
              <div className="rounded-3xl bg-white p-12 text-center shadow-sm dark:bg-[#37474F]">
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  No use cases found
                </h2>
                <p className="mt-2 text-gray-600 dark:text-gray-300">
                  There are currently no use cases available in this category.
                </p>
              </div>
            )}

            {!loading && !error && useCases.length > 0 && (
              <>
                <div className="grid items-stretch gap-6 sm:grid-cols-2 lg:grid-cols-3">
                  {paginatedUseCases.map((item) => (
                    <Link
                      key={item.id}
                      href={`/${locale}/usecases/${item.id}`}
                      className="group flex h-full min-h-[390px] flex-col overflow-hidden rounded-3xl bg-white shadow-sm transition hover:-translate-y-1 hover:shadow-xl dark:bg-[#37474F]"
                    >
                      <div className="h-52 w-full shrink-0 overflow-hidden bg-gray-200 dark:bg-[#455A64]">
                        <img
                          src={
                            item.cover_img ||
                            item.cover_image ||
                            item.image ||
                            "/img/insights/eco.webp"
                          }
                          alt={item.title || item.name || "Use case"}
                          className="h-full w-full object-cover transition duration-300 group-hover:scale-105"
                        />
                      </div>

                      <div className="flex flex-1 flex-col p-6">
                        <h2 className="mb-3 line-clamp-2 min-h-[56px] text-xl font-bold text-gray-900 dark:text-white">
                          {item.title || item.name}
                        </h2>

                        <p className="line-clamp-3 min-h-[72px] text-sm leading-6 text-gray-600 dark:text-gray-300">
                          {item.description || item.short_description || ""}
                        </p>

                        <div className="mt-auto pt-5 text-sm font-semibold text-black transition group-hover:text-green-600 dark:text-white dark:group-hover:text-green-400">
                          View Use Case →
                        </div>
                      </div>
                    </Link>
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
            )}
          </div>
        </section>
      </main>

      <Footer />
    </>
  );
};

export default CategoryUseCasesPage;