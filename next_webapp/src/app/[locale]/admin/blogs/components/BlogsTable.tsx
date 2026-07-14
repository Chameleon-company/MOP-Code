"use client";

import Link from "next/link";
import { Pencil, Trash2 } from "lucide-react";
import ImageHoverPreview from "@/components/admin/ImageHoverPreview";
import TextHoverPreview from "@/components/admin/TextHoverPreview";
function stripHtml(html: string) {
  if (!html) return "";
  return html.replace(/<[^>]*>/g, "").replace(/&nbsp;/g, " ").trim();
}

export default function BlogTable({
  data,
  locale,
  onDelete,
}: {
  data: any[];
  locale: string;
  onDelete: (id: number) => void;
}) {
  return (
    <div className="rounded-2xl bg-[#ECEAEA] p-5">
      <div className="overflow-x-auto">
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
            {data.map((item: any) => (
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
