"use client";

import Link from "next/link";
import { Pencil, Trash2 } from "lucide-react";

export default function BlogTable({ data, onDelete }: any) {
  return (
    <div className="rounded-2xl bg-[#ECEAEA] p-5">
      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b border-black/30 text-left text-sm font-semibold">
              <th className="px-3 py-4">Blog ID</th>
              <th className="px-3 py-4">Title</th>
              <th className="px-3 py-4">Sub Title</th>
              <th className="px-3 py-4">Date</th>
              <th className="px-3 py-4">Image</th>
              <th className="px-3 py-4">Description</th>
              <th className="px-3 py-4">Actions</th>
            </tr>
          </thead>

          <tbody>
            {data.map((item: any, index: number) => (
              <tr key={item.id} className="border-b border-black/10">
                <td className="px-3 py-4 text-sm">{100223100 + index}</td>

                <td className="px-3 py-4 text-sm font-medium">
                  {item.title}
                </td>

                <td className="px-3 py-4 text-sm">{item.subTitle}</td>

                <td className="px-3 py-4 text-sm">{item.date}</td>

                <td className="px-3 py-4">
                  <img
                    src={item.image}
                    alt={item.title}
                    className="h-12 w-12 rounded-lg border bg-white object-cover"
                  />
                </td>

                <td className="max-w-[300px] px-3 py-4 text-sm text-[#687280]">
                  {item.description}
                </td>

                <td className="px-3 py-4">
                  <div className="flex items-center gap-2">
                    <Link href={`blogs/edit/${item.id}`}>
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
                <td colSpan={7} className="py-8 text-center text-gray-500">
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