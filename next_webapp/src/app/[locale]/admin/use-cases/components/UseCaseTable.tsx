"use client";

import Link from "next/link";
import { Pencil, Trash2 } from "lucide-react";
import ImageHoverPreview from "@/components/admin/ImageHoverPreview";
import TextHoverPreview from "@/components/admin/TextHoverPreview";
interface UseCaseRow {
  id: number;
  title: string;
  cover_img: string | null;
  category_name: string;
  description: string | null;
}

export default function UseCaseTable({
  data,
  loading,
  locale,
  onDelete,
}: {
  data: UseCaseRow[];
  loading: boolean;
  locale: string;
  onDelete: (id: number, title: string) => void;
}) {
  return (
    <div className="rounded-2xl bg-[#ECEAEA] p-3 sm:p-5">
      {/* Mobile Card View (< 768px) */}
      <div className="space-y-3 md:hidden">
        {loading && (
          <div className="rounded-xl bg-white p-8 text-center text-sm text-[#687280]">
            Loading...
          </div>
        )}

        {!loading &&
          data.map((item) => (
            <div
              key={item.id}
              className="flex flex-col gap-3 rounded-xl border border-black/5 bg-white p-4 shadow-sm"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex min-w-0 items-center gap-3">
                  <div className="shrink-0">
                    <ImageHoverPreview
                      src={item.cover_img || "/images/category-placeholder.png"}
                      alt={item.title}
                    />
                  </div>
                  <div className="min-w-0">
                    <h4 className="truncate text-sm font-semibold text-black">
                      {item.title}
                    </h4>
                    <span className="mt-1 inline-block rounded-md bg-[#ECEAEA] px-2 py-0.5 text-xs font-medium text-[#687280]">
                      {item.category_name}
                    </span>
                  </div>
                </div>

                <div className="flex shrink-0 items-center gap-1.5">
                  <Link href={`/${locale}/admin/use-cases/edit/${item.id}`}>
                    <button
                      type="button"
                      aria-label="Edit use case"
                      className="rounded-lg bg-[#ECEAEA] p-2 text-[#1F8F50] transition hover:bg-[#DFF7E8]"
                    >
                      <Pencil size={15} />
                    </button>
                  </Link>
                  <button
                    type="button"
                    onClick={() => onDelete(item.id, item.title)}
                    aria-label="Delete use case"
                    className="rounded-lg bg-[#ECEAEA] p-2 text-red-500 transition hover:bg-red-50"
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
              </div>

              {item.description && (
                <p className="line-clamp-2 rounded-lg border border-black/5 bg-[#F9F9F9] p-2.5 text-xs text-[#687280]">
                  {item.description}
                </p>
              )}
            </div>
          ))}

        {!loading && data.length === 0 && (
          <div className="rounded-xl bg-white p-8 text-center text-sm text-[#687280]">
            No use cases found.
          </div>
        )}
      </div>

      {/* Desktop Table View (>= 768px) */}
      <div className="hidden overflow-x-auto md:block">
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b border-black/30">
              <th className="px-3 py-4 text-left text-[14px] font-semibold text-black">
                Image
              </th>
              <th className="px-3 py-4 text-left text-[14px] font-semibold text-black">
                Title
              </th>
              <th className="px-3 py-4 text-left text-[14px] font-semibold text-black">
                Category
              </th>
              <th className="px-3 py-4 text-left text-[14px] font-semibold text-black">
                Description
              </th>
              <th className="px-3 py-4 text-left text-[14px] font-semibold text-black">
                Actions
              </th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td
                  colSpan={5}
                  className="px-3 py-10 text-center text-[14px] text-[#687280]"
                >
                  Loading...
                </td>
              </tr>
            )}

            {!loading &&
              data.map((item) => (
                <tr key={item.id} className="border-b border-black/10">
                  <td className="px-3 py-4">
                    <ImageHoverPreview
                      src={item.cover_img || "/images/category-placeholder.png"}
                      alt={item.title}
                    />
                  </td>

                  <td className="px-3 py-4 text-[14px] font-medium text-black">
                    <TextHoverPreview text={item.title} />
                  </td>

                  <td className="px-3 py-4 text-[14px] text-[#687280]">
                    {item.category_name}
                  </td>

                  <td className="max-w-[280px] px-3 py-4 text-[14px] text-[#687280]">
                    <TextHoverPreview text={item.description || "—"} />
                  </td>

                  <td className="px-3 py-4">
                    <div className="flex items-center gap-2">
                      <Link href={`/${locale}/admin/use-cases/edit/${item.id}`}>
                        <button
                          type="button"
                          className="rounded-lg bg-white p-2 text-[#1F8F50] transition hover:bg-[#DFF7E8]"
                        >
                          <Pencil size={16} />
                        </button>
                      </Link>
                      <button
                        type="button"
                        onClick={() => onDelete(item.id, item.title)}
                        className="rounded-lg bg-white p-2 text-red-500 transition hover:bg-red-50"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}

            {!loading && data.length === 0 && (
              <tr>
                <td
                  colSpan={5}
                  className="px-3 py-10 text-center text-[14px] text-[#687280]"
                >
                  No use cases found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
