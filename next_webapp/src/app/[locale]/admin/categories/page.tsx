import Link from "next/link";
import { FolderOpen, Plus, Pencil, Trash2 } from "lucide-react";

const categories = [
  {
    id: 1,
    title: "Transport",
    description: "Transport related open data and services.",
    image: "/images/category-placeholder.png",
  },
  {
    id: 2,
    title: "Environment",
    description: "Environment related insights and reports.",
    image: "/images/category-placeholder.png",
  },
  {
    id: 3,
    title: "Health",
    description: "Health and wellbeing data resources.",
    image: "/images/category-placeholder.png",
  },
];

export default function CategoriesPage({
  params,
}: {
  params: { locale: string };
}) {
  return (
    <div>
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-[40px] font-semibold leading-[48px] text-[#2DBE6C]">
            Categories
          </h1>
          <p className="mt-2 text-[16px] leading-[24px] text-[#687280]">
            View all current categories here.
          </p>
        </div>

        <Link
          href={`/${params.locale}/admin/categories/add`}
          className="inline-flex items-center gap-2 rounded-lg bg-[#1F8F50] px-5 py-3 text-[14px] font-medium text-white transition hover:bg-[#2DBE6C]"
        >
          <Plus size={18} />
          Add New
        </Link>
      </div>

      {/* <div className="mb-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="rounded-2xl bg-[#ECEAEA] px-5 py-6">
          <div className="mb-3 flex justify-center text-[#2DBE6C]">
            <FolderOpen size={30} />
          </div>
          <h3 className="text-center text-[32px] font-semibold text-black">
            {categories.length}
          </h3>
          <p className="mt-2 text-center text-[14px] text-black">
            Total Categories
          </p>
        </div>
      </div> */}

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
                  Description
                </th>
                <th className="px-3 py-4 text-left text-[14px] font-semibold text-black">
                  Actions
                </th>
              </tr>
            </thead>

            <tbody>
              {categories.map((category) => (
                <tr key={category.id} className="border-b border-black/10">
                  <td className="px-3 py-4">
                    <img
                      src={category.image}
                      alt={category.title}
                      className="h-14 w-14 rounded-lg object-cover border border-gray-300 bg-white"
                    />
                  </td>

                  <td className="px-3 py-4 text-[14px] font-medium text-black">
                    {category.title}
                  </td>

                  <td className="px-3 py-4 text-[14px] text-[#687280]">
                    {category.description}
                  </td>

                  <td className="px-3 py-4">
                    <div className="flex items-center gap-2">
                      <button className="rounded-lg bg-white p-2 text-[#1F8F50] transition hover:bg-[#DFF7E8]">
                        <Pencil size={16} />
                      </button>
                      <button className="rounded-lg bg-white p-2 text-red-500 transition hover:bg-red-50">
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}

              {categories.length === 0 && (
                <tr>
                  <td
                    colSpan={4}
                    className="px-3 py-8 text-center text-[14px] text-[#687280]"
                  >
                    No categories found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}