"use client";

import Link from "next/link";
import { useState } from "react";
import { Plus, Search, Filter } from "lucide-react";
import BlogTable from "./components/BlogsTable";

export default function BlogsPage({ params }: any) {
  const [data, setData] = useState([
    {
      id: 1,
      title: "How Data Supports Better City Planning",
      subTitle: "Using data to improve community planning",
      date: "2026-04-27",
      description:
        "Explore how data-driven insights help improve urban planning and community services.",
      image: "/images/category-placeholder.png",
    },
    {
      id: 2,
      title: "Community Impact Through Digital Innovation",
      subTitle: "Technology for better community outcomes",
      date: "2026-04-28",
      description:
        "A short blog about using digital tools to support better social and community outcomes.",
      image: "/images/category-placeholder.png",
    },
    {
      id: 3,
      title: "Education and Technology",
      subTitle: "Modern tools for learning support",
      date: "2026-04-29",
      description:
        "Discussing how modern technology can support learning and teaching outcomes.",
      image: "/images/category-placeholder.png",
    },
  ]);

  const [search, setSearch] = useState("");
  const [selectedDate, setSelectedDate] = useState("");

  const handleDelete = (id: number) => {
    if (confirm("Are you sure you want to delete this blog?")) {
      setData(data.filter((item) => item.id !== id));
    }
  };

  const filteredData = data.filter((item) => {
    const matchesSearch =
      item.title.toLowerCase().includes(search.toLowerCase()) ||
      item.subTitle.toLowerCase().includes(search.toLowerCase()) ||
      item.description.toLowerCase().includes(search.toLowerCase());

    const matchesDate = !selectedDate || item.date === selectedDate;

    return matchesSearch && matchesDate;
  });

  return (
    <div>
      {/* header */}
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-[40px] font-semibold text-[#2DBE6C]">Blogs</h1>
          <p className="mt-2 text-[16px] text-[#687280]">
            Manage and organize your blogs
          </p>
        </div>

        <Link
          href={`/${params.locale}/admin/blogs/add`}
          className="flex items-center gap-2 rounded-lg bg-[#1F8F50] px-5 py-3 text-white hover:bg-[#2DBE6C]"
        >
          <Plus size={18} />
          Add New
        </Link>
      </div>

      {/* search + date filter */}
      <div className="mb-6 flex flex-col gap-3 md:flex-row">
        <div className="flex flex-1 items-center gap-2 rounded-xl border border-[#CFEFD9] bg-[#F8FFFA] px-4 py-3">
          <Search size={18} className="text-[#1F8F50]" />
          <input
            placeholder="Search blogs..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-transparent text-sm outline-none"
          />
        </div>

        <div className="flex items-center gap-2 rounded-xl border border-[#CFEFD9] bg-[#F8FFFA] px-4 py-3">
          <Filter size={18} className="text-[#1F8F50]" />
          <input
            type="date"
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
            className="bg-transparent text-sm outline-none"
          />

          {selectedDate && (
            <button
              type="button"
              onClick={() => setSelectedDate("")}
              className="text-sm text-[#1F8F50] hover:underline"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* table */}
      <BlogTable data={filteredData} onDelete={handleDelete} />
    </div>
  );
}