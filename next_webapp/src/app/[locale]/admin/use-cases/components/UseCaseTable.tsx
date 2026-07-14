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
    <div className="rounded-2xl bg-[#ECEAEA] p-5">
      <div className="overflow-x-auto">
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

                  <td className="px-3 py-4 text-[14px] text-[#687280] max-w-[280px]">
  <TextHoverPreview text={item.description || "—"} />
</td>

                  <td className="px-3 py-4">
                    <div className="flex items-center gap-2">
                      <Link href={`/${locale}/admin/use-cases/edit/${item.id}`}>
                        <button className="rounded-lg bg-white p-2 text-[#1F8F50] transition hover:bg-[#DFF7E8]">
                          <Pencil size={16} />
                        </button>
                      </Link>
                      <button
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
