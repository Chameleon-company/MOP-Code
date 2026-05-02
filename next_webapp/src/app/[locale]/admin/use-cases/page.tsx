"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { Plus, Search, Filter, ChevronLeft, ChevronRight } from "lucide-react";
import UseCaseTable from "./components/UseCaseTable";

function getAuthHeaders() {
  const user = JSON.parse(localStorage.getItem("user") || "{}");
  const userId = user.userId ?? user.id ?? "";
  const roleId = user.roleId ?? user.role_id ?? "";
  const token = user.token ?? "";
  return {
    "x-user-id": String(userId),
    "x-user-role-id": String(roleId),
    "x-user-role": user.roleName ?? user.role_name ?? "",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

const PAGE_SIZE = 10;

export default function UseCasesPage() {
  const { locale } = useParams() as { locale: string };

  const [usecases, setUsecases] = useState<any[]>([]);
  const [categories, setCategories] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("All");
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);

  // Fetch all categories once for the dropdown
  useEffect(() => {
    fetch("/api/categories?pageSize=100", { headers: getAuthHeaders() })
      .then((r) => r.json())
      .then((json) => { if (json.success) setCategories(json.data || []); })
      .catch(() => {});
  }, []);

  // Re-fetch use cases whenever page, search, or category filter changes
  useEffect(() => {
    fetchUseCases();
  }, [page, search, selectedCategory]);

  async function fetchUseCases() {
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams({
        page: String(page),
        pageSize: String(PAGE_SIZE),
      });
      if (search) params.set("search", search);
      if (selectedCategory !== "All") params.set("category_id", selectedCategory);

      const res = await fetch(`/api/usecases?${params}`, { headers: getAuthHeaders() });
      const json = await res.json();
      if (json.success) {
        setUsecases(json.data || []);
        setTotalPages(json.pagination?.totalPages ?? 1);
        setTotal(json.pagination?.total ?? 0);
      } else {
        setError(json.message || json.error || "Failed to load use cases.");
      }
    } catch {
      setError("Failed to load data.");
    } finally {
      setLoading(false);
    }
  }

  function handleSearch(value: string) {
    setSearch(value);
    setPage(1);
  }

  function handleCategoryChange(value: string) {
    setSelectedCategory(value);
    setPage(1);
  }

  async function handleDelete(id: number, title: string) {
    if (!confirm(`Delete "${title}"? This cannot be undone.`)) return;
    try {
      const res = await fetch(`/api/usecases/${id}`, {
        method: "DELETE",
        headers: getAuthHeaders(),
      });
      const json = await res.json();
      if (!json.success) {
        alert(json.message || json.error || "Failed to delete use case.");
        return;
      }
      if (usecases.length === 1 && page > 1) {
        setPage((p) => p - 1);
      } else {
        fetchUseCases();
      }
    } catch {
      alert("Failed to delete use case.");
    }
  }

  const categoryMap = Object.fromEntries(categories.map((c) => [c.id, c.category_name]));
  const displayData = usecases.map((u) => ({
    ...u,
    category_name: categoryMap[u.category_id] ?? "—",
  }));

  const rangeStart = total === 0 ? 0 : (page - 1) * PAGE_SIZE + 1;
  const rangeEnd = Math.min(page * PAGE_SIZE, total);

  return (
    <div>
      {/* Header */}
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-[40px] font-semibold text-[#2DBE6C]">Use Cases</h1>
          <p className="mt-2 text-[16px] text-[#687280]">
            Manage and organize your use cases
          </p>
        </div>
        <Link
          href={`/${locale}/admin/use-cases/add`}
          className="inline-flex items-center gap-2 rounded-lg bg-[#1F8F50] px-5 py-3 text-[14px] font-medium text-white transition hover:bg-[#2DBE6C]"
        >
          <Plus size={18} />
          Add New
        </Link>
      </div>

      {error && (
        <div className="mb-4 rounded-xl bg-red-50 px-4 py-3 text-sm text-red-600">
          {error}
        </div>
      )}

      {/* Search + Filter */}
      <div className="mb-6 flex flex-col gap-3 md:flex-row">
        <div className="flex flex-1 items-center gap-2 rounded-xl border border-[#CFEFD9] bg-[#F8FFFA] px-4 py-3">
          <Search size={18} className="text-[#1F8F50]" />
          <input
            placeholder="Search use cases..."
            value={search}
            onChange={(e) => handleSearch(e.target.value)}
            className="w-full bg-transparent text-sm outline-none"
          />
        </div>
        <div className="flex items-center gap-2 rounded-xl border border-[#CFEFD9] bg-[#F8FFFA] px-4 py-3">
          <Filter size={18} className="text-[#1F8F50]" />
          <select
            value={selectedCategory}
            onChange={(e) => handleCategoryChange(e.target.value)}
            className="bg-transparent text-sm outline-none"
          >
            <option value="All">All Categories</option>
            {categories.map((c) => (
              <option key={c.id} value={String(c.id)}>
                {c.category_name}
              </option>
            ))}
          </select>
        </div>
      </div>

      <UseCaseTable
        data={displayData}
        loading={loading}
        locale={locale}
        onDelete={handleDelete}
      />

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="mt-6 flex items-center justify-between">
          <p className="text-sm text-[#687280]">
            Showing {rangeStart}–{rangeEnd} of {total}
          </p>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage((p) => p - 1)}
              disabled={page === 1}
              className="inline-flex items-center gap-1 rounded-lg border border-[#CFEFD9] bg-white px-3 py-2 text-sm text-[#1F8F50] transition hover:bg-[#DFF7E8] disabled:cursor-not-allowed disabled:opacity-40"
            >
              <ChevronLeft size={16} />
              Previous
            </button>
            <span className="text-sm text-[#687280]">
              Page {page} of {totalPages}
            </span>
            <button
              onClick={() => setPage((p) => p + 1)}
              disabled={page === totalPages}
              className="inline-flex items-center gap-1 rounded-lg border border-[#CFEFD9] bg-white px-3 py-2 text-sm text-[#1F8F50] transition hover:bg-[#DFF7E8] disabled:cursor-not-allowed disabled:opacity-40"
            >
              Next
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
