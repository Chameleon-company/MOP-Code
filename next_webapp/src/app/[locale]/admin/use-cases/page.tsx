// export default function UseCasesPage() {
//   return (
//     <div>
//       <h1 className="text-[40px] font-semibold leading-[48px] text-[#2DBE6C]">
//         Use Cases
//       </h1>
//     </div>
//   );
// }
"use client";

import Link from "next/link";
import { useState } from "react";
import { Plus, Search, Filter } from "lucide-react";
import { useParams } from "next/navigation"; 
import UseCaseTable from "./components/UseCaseTable";

export default function UseCasesPage({ params }: any) {
  
  const [data, setData] = useState([
    {
      id: 1,
      title: "Business and Economy",
      description:
        "Analyze market trends and financial data to optimize business strategies.",
      category: "Category 1",
      image: "/images/category-placeholder.png",
    },
    {
      id: 2,
      title: "Community and Social Impact",
      description:
        "Develop initiatives to enhance social welfare and community outcomes.",
      category: "Category 2",
      image: "/images/category-placeholder.png",
    },
    {
      id: 3,
      title: "Education and Teaching",
      description:
        "Leverage AI tools to improve learning and personalize education.",
      category: "Category 1",
      image: "/images/category-placeholder.png",
    },
    {
      id: 4,
      title: "Environmental Sustainability",
      description:
        "Implement eco-friendly strategies to protect natural resources.",
      category: "Category 3",
      image: "/images/category-placeholder.png",
    },
  ]);

  const [search, setSearch] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("All");

  const handleDelete = (id: number) => {
    if (confirm("Are you sure you want to delete this use case?")) {
      setData(data.filter((item) => item.id !== id));
    }
  };

  // Filtering logic
  const filteredData = data.filter((item) => {
    const matchesSearch =
      item.title.toLowerCase().includes(search.toLowerCase()) ||
      item.description.toLowerCase().includes(search.toLowerCase());

    const matchesCategory =
      selectedCategory === "All" ||
      item.category === selectedCategory;

    return matchesSearch && matchesCategory;
  });

  return (
    <div>

      {/* header */}
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-[40px] font-semibold text-[#2DBE6C]">
            Use Cases
          </h1>
          <p className="mt-2 text-[16px] text-[#687280]">
            Manage and organize your use cases
          </p>
        </div>

        <Link
          href={`/${params.locale}/admin/use-cases/add`}
          className="flex items-center gap-2 rounded-lg bg-[#1F8F50] px-5 py-3 text-white hover:bg-[#2DBE6C]"
        >
          <Plus size={18} />
          Add New
        </Link>
      </div>

      {/* saerch + filter  */}
      <div className="mb-6 flex flex-col md:flex-row gap-3">

        <div className="flex flex-1 items-center gap-2 rounded-xl border border-[#CFEFD9] bg-[#F8FFFA] px-4 py-3">
          <Search size={18} className="text-[#1F8F50]" />
          <input
            placeholder="Search use cases..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-transparent outline-none text-sm"
          />
        </div>

        <div className="flex items-center gap-2 rounded-xl border border-[#CFEFD9] bg-[#F8FFFA] px-4 py-3">
          <Filter size={18} className="text-[#1F8F50]" />
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="bg-transparent outline-none text-sm"
          >
            <option value="All">All Categories</option>
            <option value="Category 1">Category 1</option>
            <option value="Category 2">Category 2</option>
            <option value="Category 3">Category 3</option>
          </select>
        </div>

      </div>

      {/* tables */}
      <UseCaseTable data={filteredData} onDelete={handleDelete} />

    </div>
  );
}