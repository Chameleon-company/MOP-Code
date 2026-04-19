"use client";

import Link from "next/link";
import { Pencil, Trash2 } from "lucide-react";

export default function UseCaseTable({ data, onDelete }: any) {
  return (
    <div className="rounded-2xl bg-[#ECEAEA] p-5">
      <div className="overflow-x-auto">
        <table className="w-full border-collapse">

          {/* Header */}
          <thead>
            <tr className="border-b border-black/30 text-left text-sm font-semibold">
              <th className="px-3 py-4">Serial Number</th>
              <th className="px-3 py-4">Title</th>
              <th className="px-3 py-4">Cover Image</th>
              <th className="px-3 py-4">Category</th>
              <th className="px-3 py-4">Description</th>
              <th className="px-3 py-4">Actions</th>
            </tr>
          </thead>

          {/* Body */}
          <tbody>
            {data.map((item: any, index: number) => (
              <tr key={item.id} className="border-b border-black/10">

                {/* Serial number */}
                <td className="px-3 py-4 text-sm">
                  {100123100 + index}
                </td>

                {/* title */}
                <td className="px-3 py-4 text-sm font-medium">
                  {item.title}
                </td>

                {/* Image */}
                <td className="px-3 py-4">
                  <img
                    src={item.image}
                    alt="cover"
                    className="h-12 w-12 rounded-lg object-cover border bg-white"
                  />
                </td>

                {/* Category */}
                <td className="px-3 py-4 text-sm">
                  {item.category}
                </td>

                {/* Description */}
                <td className="px-3 py-4 text-sm text-[#687280] max-w-[300px]">
                  {item.description}
                </td>

                {/* actions */}
                <td className="px-3 py-4">
                  <div className="flex items-center gap-2">

                    <Link href={`use-cases/edit/${item.id}`}>
                      <button className="rounded-lg bg-white p-2 text-[#1F8F50] hover:bg-[#DFF7E8]">
                        <Pencil size={16} />
                      </button>
                    </Link>

                    <button
                      onClick={() => onDelete(item.id)}
                      className="rounded-lg bg-white p-2 text-red-500 hover:bg-red-50"
                    >
                      <Trash2 size={16} />
                    </button>

                  </div>
                </td>

              </tr>
            ))}

            {/* empty state */}
            {data.length === 0 && (
              <tr>
                <td colSpan={6} className="text-center py-8 text-gray-500">
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