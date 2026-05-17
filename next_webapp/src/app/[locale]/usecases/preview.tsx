// "use client";

// import { useParams } from "next/navigation";
// import Link from "next/link";
// import Image from "next/image";
// import React from "react";
// import { ArrowRight, ChevronLeft, ChevronRight, FileText } from "lucide-react";
// import { CaseStudy } from "../../types";

// const UseCaseCard: React.FC<{ study: CaseStudy }> = ({ study }) => {
//   const params = useParams<{ locale: string }>();
//   const locale = params?.locale ?? "en";

//   return (
//     <Link
//       href={`/${locale}/usecases/${study.id}`}
//       className="group block overflow-hidden rounded-[24px] border border-gray-200 bg-white shadow-sm transition duration-300 hover:-translate-y-1 hover:border-green-400 hover:shadow-xl dark:border-gray-700 dark:bg-gray-800"
//     >
//       <div className="relative h-44 w-full bg-gray-100 dark:bg-gray-700">
//         {study.image ? (
//           <Image
//             src={study.image}
//             alt={study.title}
//             fill
//             className="object-cover transition duration-300 group-hover:scale-105"
//           />
//         ) : (
//           <div className="flex h-full w-full items-center justify-center text-green-600 dark:text-green-300">
//             <FileText size={34} />
//           </div>
//         )}
//       </div>

//       <div className="p-5">
//         <h3 className="mb-3 text-xl font-bold leading-snug text-gray-900 dark:text-white">
//           {study.title}
//         </h3>

//         <p className="mb-5 min-h-[72px] text-sm leading-6 text-gray-600 dark:text-gray-300">
//           {study.description}
//         </p>

//         <div className="mb-5 flex flex-wrap gap-2">
//           {study.tags?.slice(0, 3).map((tag) => (
//             <span
//               key={tag}
//               className="rounded-full border border-gray-200 bg-gray-50 px-3 py-1 text-xs font-medium text-gray-700 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-200"
//             >
//               {tag}
//             </span>
//           ))}
//         </div>

//         <div className="flex items-center justify-between border-t border-gray-100 pt-4 dark:border-gray-700">
//           <span className="text-sm font-semibold text-gray-700 dark:text-gray-200">
//             View overview
//           </span>
//           <ArrowRight
//             className="text-green-600 transition group-hover:translate-x-1"
//             size={18}
//           />
//         </div>
//       </div>
//     </Link>
//   );
// };

// interface Props {
//   caseStudies: CaseStudy[];
//   page: number;
//   totalPages: number;
//   total: number;
//   onPageChange: (page: number) => void;
// }

// const PreviewComponent: React.FC<Props> = ({
//   caseStudies,
//   page,
//   totalPages,
//   total,
//   onPageChange,
// }) => {
//   if (!caseStudies.length) {
//     return (
//       <section className="rounded-[28px] border border-dashed border-gray-300 bg-white px-6 py-16 text-center shadow-sm dark:border-gray-600 dark:bg-gray-800">
//         <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-full bg-green-50 text-green-600 dark:bg-green-900/30 dark:text-green-300">
//           <FileText size={28} />
//         </div>
//         <h2 className="mb-3 text-2xl font-bold text-gray-900 dark:text-white">
//           No use cases found
//         </h2>
//         <p className="mx-auto max-w-xl text-base leading-7 text-gray-600 dark:text-gray-300">
//           Try another keyword or search mode. You can also reset the filters to
//           explore all available use cases again.
//         </p>
//       </section>
//     );
//   }

//   return (
//     <div className="space-y-6">
//       <section className="rounded-[28px] border border-gray-200 bg-white p-5 shadow-sm dark:border-gray-700 dark:bg-gray-800 sm:p-6">
//         <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
//           <div>
//             <p className="mb-2 text-sm font-semibold uppercase tracking-[0.2em] text-green-600 dark:text-green-300">
//               Case Study Gallery
//             </p>
//             <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
//               Tiled showcase of use cases
//             </h2>
//             <p className="mt-2 text-sm text-gray-600 dark:text-gray-300">
//               Showing {caseStudies.length} of {total} use cases
//             </p>
//           </div>

//           <div className="inline-flex items-center rounded-full bg-gray-100 px-4 py-2 text-sm font-semibold text-gray-700 dark:bg-gray-700 dark:text-white">
//             Page {page} of {totalPages || 1}
//           </div>
//         </div>
//       </section>

//       <section className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
//         {caseStudies.map((study) => (
//           <UseCaseCard key={study.id} study={study} />
//         ))}
//       </section>

//       {totalPages > 1 && (
//         <div className="flex items-center justify-center gap-3">
//           <button
//             onClick={() => onPageChange(page - 1)}
//             disabled={page === 1}
//             className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-gray-200 bg-white text-gray-700 transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-gray-600 dark:bg-gray-800 dark:text-white dark:hover:bg-gray-700"
//           >
//             <ChevronLeft size={18} />
//           </button>

//           <span className="rounded-full bg-green-50 px-4 py-2 text-sm font-semibold text-green-700 dark:bg-green-900/30 dark:text-green-300">
//             {page}
//           </span>

//           <button
//             onClick={() => onPageChange(page + 1)}
//             disabled={page === totalPages}
//             className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-gray-200 bg-white text-gray-700 transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-gray-600 dark:bg-gray-800 dark:text-white dark:hover:bg-gray-700"
//           >
//             <ChevronRight size={18} />
//           </button>
//         </div>
//       )}
//     </div>
//   );
// };

// export default PreviewComponent;
"use client";

import { useParams } from "next/navigation";
import Link from "next/link";
import Image from "next/image";
import React from "react";
import { ArrowRight, FileText } from "lucide-react";
import { CaseStudy } from "../../types";
import Pagination from "@/components/Pagination";

const UseCaseCard: React.FC<{ study: CaseStudy }> = ({ study }) => {
  const params = useParams<{ locale: string }>();
  const locale = params?.locale ?? "en";

  return (
    <Link
      href={`/${locale}/usecases/${study.id}`}
      className="group flex h-full flex-col overflow-hidden rounded-[24px] border border-gray-200 bg-white shadow-sm transition duration-300 hover:-translate-y-1 hover:border-green-400 hover:shadow-xl dark:border-gray-700 dark:bg-gray-800"
    >
      <div className="relative h-44 w-full shrink-0 overflow-hidden bg-gray-100 dark:bg-gray-700">
        {study.image ? (
          <Image
            src={study.image}
            alt={study.title}
            fill
            className="object-cover transition duration-300 group-hover:scale-105"
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center text-green-600 dark:text-green-300">
            <FileText size={34} />
          </div>
        )}
      </div>

      <div className="flex flex-1 flex-col p-5">
        <h3 className="mb-3 min-h-[56px] text-xl font-bold leading-snug text-gray-900 dark:text-white">
          {study.title}
        </h3>

        <p className="mb-5 min-h-[96px] text-sm leading-6 text-gray-600 dark:text-gray-300">
          {study.description}
        </p>

        <div className="mb-5 flex min-h-[34px] flex-wrap gap-2">
          {study.tags?.slice(0, 3).map((tag) => (
            <span
              key={tag}
              className="rounded-full border border-gray-200 bg-gray-50 px-3 py-1 text-xs font-medium text-gray-700 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-200"
            >
              {tag}
            </span>
          ))}
        </div>

        <div className="mt-auto flex items-center justify-between border-t border-gray-100 pt-4 dark:border-gray-700">
          <span className="text-sm font-semibold text-gray-700 dark:text-gray-200">
            View overview
          </span>
          <ArrowRight
            className="text-green-600 transition group-hover:translate-x-1"
            size={18}
          />
        </div>
      </div>
    </Link>
  );
};

interface Props {
  caseStudies: CaseStudy[];
  page: number;
  totalPages: number;
  total: number;
  onPageChange: (page: number) => void;
}

const PreviewComponent: React.FC<Props> = ({
  caseStudies,
  page,
  totalPages,
  total,
  onPageChange,
}) => {
  if (!caseStudies.length) {
    return (
      <section className="rounded-[28px] border border-dashed border-gray-300 bg-white px-6 py-16 text-center shadow-sm dark:border-gray-600 dark:bg-gray-800">
        <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-full bg-green-50 text-green-600 dark:bg-green-900/30 dark:text-green-300">
          <FileText size={28} />
        </div>
        <h2 className="mb-3 text-2xl font-bold text-gray-900 dark:text-white">
          No use cases found
        </h2>
        <p className="mx-auto max-w-xl text-base leading-7 text-gray-600 dark:text-gray-300">
          Try another keyword or search mode. You can also reset the filters to
          explore all available use cases again.
        </p>
      </section>
    );
  }

  return (
    <div className="space-y-6">
      <section className="rounded-[28px] border border-gray-200 bg-white p-5 shadow-sm dark:border-gray-700 dark:bg-gray-800 sm:p-6">
        <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="mb-2 text-sm font-semibold uppercase tracking-[0.2em] text-green-600 dark:text-green-300">
              Case Study Gallery
            </p>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
              Tiled showcase of use cases
            </h2>
            <p className="mt-2 text-sm text-gray-600 dark:text-gray-300">
              Showing {caseStudies.length} of {total} use cases
            </p>
          </div>

          <div className="inline-flex items-center rounded-full bg-gray-100 px-4 py-2 text-sm font-semibold text-gray-700 dark:bg-gray-700 dark:text-white">
            Page {page} of {totalPages || 1}
          </div>
        </div>
      </section>

      <section className="grid items-stretch gap-6 sm:grid-cols-2 xl:grid-cols-3">
        {caseStudies.map((study) => (
          <UseCaseCard key={study.id} study={study} />
        ))}
      </section>

    <Pagination page={page} totalPages={totalPages} onPageChange={onPageChange} />
    </div>
  );
};

export default PreviewComponent;