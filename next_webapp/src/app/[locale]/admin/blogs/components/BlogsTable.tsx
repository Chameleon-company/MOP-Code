"use client";

import Link from "next/link";
import { Pencil, Trash2 } from "lucide-react";
import ImageHoverPreview from "@/components/admin/ImageHoverPreview";
import TextHoverPreview from "@/components/admin/TextHoverPreview";
function stripHtml(html: string) {
  if (!html) return "";
  return html.replace(/<[^>]*>/g, "").replace(/&nbsp;/g, " ").trim();
}

interface BlogItem {
  id: number;
  cover_img?: string | null;
  title: string;
  description?: string | null;
  published_date?: string | null;
  content?: string | null;
  created_by_name?: string | null;
}

export default function BlogTable({
  data,
  locale,
  onDelete,
}: {
  data: BlogItem[];
  locale: string;
  onDelete: (id: number) => void;
}) {
  return (
    <div className="rounded-2xl bg-[#ECEAEA] p-3 sm:p-5">
      {/* Mobile Card View (< 768px) */}
      <div className="space-y-3 md:hidden">
        {data.map((item) => (
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
                  <div className="flex items-center gap-2">
                    <span className="text-[11px] font-semibold text-[#687280]">
                      #{item.id}
                    </span>
                    <span className="text-[11px] text-[#687280]">
                      {item.published_date || "—"}
                    </span>
                  </div>
                  <h4 className="truncate text-sm font-semibold text-black">
                    {item.title}
                  </h4>
                </div>
              </div>

              <div className="flex shrink-0 items-center gap-1.5">
                <Link href={`/${locale}/admin/blogs/edit/${item.id}`}>
                  <button
                    type="button"
                    aria-label="Edit blog"
                    className="rounded-lg bg-[#ECEAEA] p-2 text-[#1F8F50] transition hover:bg-[#DFF7E8]"
                  >
                    <Pencil size={15} />
                  </button>
                </Link>
                <button
                  type="button"
                  onClick={() => onDelete(item.id)}
                  aria-label="Delete blog"
                  className="rounded-lg bg-[#ECEAEA] p-2 text-red-500 transition hover:bg-red-50"
                >
                  <Trash2 size={15} />
                </button>
              </div>
            </div>

            {item.description && (
              <p className="line-clamp-2 text-xs text-[#687280]">
                {item.description}
              </p>
            )}

            <div className="flex items-center justify-between border-t border-black/5 pt-2 text-xs text-[#687280]">
              <span>
                By:{" "}
                <span className="font-medium text-black">
                  {item.created_by_name ?? "—"}
                </span>
              </span>
            </div>
          </div>
        ))}

        {data.length === 0 && (
          <div className="rounded-xl bg-white p-8 text-center text-sm text-[#687280]">
            No blogs found.
          </div>
        )}
      </div>

      {/* Desktop Table View (>= 768px) */}
      <div className="hidden overflow-x-auto md:block">
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b border-black/30 text-left text-sm font-semibold">
              <th className="px-3 py-4">ID</th>
              <th className="px-3 py-4">Cover</th>
              <th className="px-3 py-4">Title</th>
              <th className="px-3 py-4">Description</th>
              <th className="px-3 py-4">Published</th>
              <th className="px-3 py-4">Content preview</th>
              <th className="px-3 py-4">Created by</th>
              <th className="px-3 py-4">Actions</th>
            </tr>
          </thead>

          <tbody>
            {data.map((item) => (
              <tr key={item.id} className="border-b border-black/10">
                <td className="px-3 py-4 text-sm text-[#687280]">{item.id}</td>

                <td className="px-3 py-4">
                  <ImageHoverPreview
                    src={item.cover_img || "/images/category-placeholder.png"}
                    alt={item.title}
                  />
                </td>

                <td className="px-3 py-4 text-sm font-medium">{item.title}</td>

                <td className="max-w-[220px] px-3 py-4 text-sm text-[#687280]">
                  <TextHoverPreview text={item.description || "—"} />
                </td>

                <td className="px-3 py-4 text-sm text-[#687280]">
                  {item.published_date || "—"}
                </td>

                <td className="max-w-[260px] px-3 py-4 text-sm text-[#687280]">
                  <TextHoverPreview text={stripHtml(item.content) || "—"} />
                </td>

                <td className="px-3 py-4 text-sm text-[#687280]">
                  {item.created_by_name ?? "—"}
                </td>

                <td className="px-3 py-4">
                  <div className="flex items-center gap-2">
                    <Link href={`/${locale}/admin/blogs/edit/${item.id}`}>
                      <button
                        type="button"
                        className="rounded-lg bg-white p-2 text-[#1F8F50] hover:bg-[#DFF7E8]"
                      >
                        <Pencil size={16} />
                      </button>
                    </Link>

                    <button
                      type="button"
                      onClick={() => onDelete(item.id)}
                      className="rounded-lg bg-white p-2 text-red-500 hover:bg-red-50"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                </td>
              </tr>
            ))}

            {data.length === 0 && (
              <tr>
                <td colSpan={8} className="py-8 text-center text-gray-500">
                  No blogs found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
