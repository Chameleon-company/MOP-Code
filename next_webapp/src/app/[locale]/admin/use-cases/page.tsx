"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { Plus, Search, Filter } from "lucide-react";
import UseCaseTable from "./components/UseCaseTable";
import ConfirmModal from "@/components/admin/ConfirmModal";
import AdminToast from "@/components/admin/AdminToast";
import Pagination from "@/components/Pagination";

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
  const [deleteTarget, setDeleteTarget] = useState<{ id: number; title: string } | null>(null);
const [toast, setToast] = useState<{ message: string; type: "success" | "error" } | null>(null);

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

  function handleDelete(id: number, title: string) {
    setDeleteTarget({ id, title });
  }
  
  async function confirmDeleteUseCase() {
    if (!deleteTarget) return;
  
    try {
      const res = await fetch(`/api/usecases/${deleteTarget.id}`, {
        method: "DELETE",
        headers: getAuthHeaders(),
      });
  
      const json = await res.json();
  
      if (!json.success) {
        setToast({
          message: json.message || json.error || "Failed to delete use case.",
          type: "error",
        });
        return;
      }
  
      setToast({ message: "Use case deleted successfully.", type: "success" });
      setDeleteTarget(null);
  
      if (usecases.length === 1 && page > 1) {
        setPage((p) => p - 1);
      } else {
        fetchUseCases();
      }
    } catch {
      setToast({ message: "Failed to delete use case.", type: "error" });
    }
  }

  const categoryMap = Object.fromEntries(categories.map((c) => [c.id, c.category_name]));
  const displayData = usecases.map((u) => ({
    ...u,
    category_name: categoryMap[u.category_id] ?? "—",
  }));

  return (
    <div>
      {/* Header */}
      <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-emerald-500 sm:text-3xl md:text-[40px]">
            Use Cases
          </h1>
          <p className="mt-1 text-xs text-[#687280] sm:text-sm md:text-[16px]">
            Manage and organize your use cases
          </p>
        </div>
        <Link
          href={`/${locale}/admin/use-cases/add`}
          className="inline-flex shrink-0 items-center justify-center gap-2 rounded-lg bg-emerald-600 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-emerald-500 sm:px-5 sm:py-3"
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
      <div className="mb-6 flex flex-col gap-3 sm:flex-row">
        <div className="flex flex-1 items-center gap-2 rounded-xl border border-[#CFEFD9] bg-[#F8FFFA] px-3.5 py-2.5 sm:px-4 sm:py-3">
          <Search size={18} className="shrink-0 text-[#1F8F50]" />
          <input
            placeholder="Search use cases..."
            value={search}
            onChange={(e) => handleSearch(e.target.value)}
            className="w-full min-w-0 bg-transparent text-sm outline-none"
          />
        </div>
        <div className="flex items-center gap-2 rounded-xl border border-[#CFEFD9] bg-[#F8FFFA] px-3.5 py-2.5 sm:px-4 sm:py-3">
          <Filter size={18} className="shrink-0 text-[#1F8F50]" />
          <select
            value={selectedCategory}
            onChange={(e) => handleCategoryChange(e.target.value)}
            className="w-full bg-transparent text-sm outline-none sm:w-auto"
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

      <Pagination
        page={page}
        totalPages={totalPages}
        onPageChange={setPage}
        total={total}
        pageSize={PAGE_SIZE}
        variant="admin"
      />
            <ConfirmModal
        open={!!deleteTarget}
        title="Delete Use Case"
        message={`Are you sure you want to delete "${deleteTarget?.title}"? This action cannot be undone.`}
        confirmText="Delete"
        cancelText="Cancel"
        isDanger
        onConfirm={confirmDeleteUseCase}
        onCancel={() => setDeleteTarget(null)}
      />

      {toast && (
        <AdminToast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}
    </div>
  );
}
