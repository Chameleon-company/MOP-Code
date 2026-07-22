"use client";

import Link from "next/link";
import { use, useState, useEffect, useCallback } from "react";
import { Plus, Search, Filter } from "lucide-react";
import BlogTable from "./components/BlogsTable";
import ConfirmModal from "@/components/admin/ConfirmModal";
import AdminToast from "@/components/admin/AdminToast";
import Pagination from "@/components/Pagination";

function authHeaders() {
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

export default function BlogsPage({ params }: { params: Promise<{ locale: string }> }) {
  const { locale } = use(params);

  const [blogs, setBlogs] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [selectedDate, setSelectedDate] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<{ id: number } | null>(null);
  const [toast, setToast] = useState<{ message: string; type: "success" | "error" } | null>(null);

  const fetchBlogs = useCallback(async (searchTerm: string, dateFilter: string, currentPage: number) => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        page: String(currentPage),
        pageSize: String(PAGE_SIZE),
      });
      if (searchTerm) { params.set("search", searchTerm); params.set("search_by", "all"); }
      if (dateFilter) { params.set("date_from", dateFilter); params.set("date_to", dateFilter); }

      const res = await fetch(`/api/blogs?${params}`, { headers: authHeaders() });
      const json = await res.json();
      if (json.success) {
        setBlogs(json.data ?? []);
        setTotal(json.pagination?.total ?? 0);
        setTotalPages(json.pagination?.totalPages ?? 1);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchBlogs(search, selectedDate, page);
  }, [search, selectedDate, page, fetchBlogs]);

  const handleSearchChange = (value: string) => {
    setSearch(value);
    setPage(1);
  };

  const handleDateChange = (value: string) => {
    setSelectedDate(value);
    setPage(1);
  };

  const handleDelete = (id: number) => {
    setDeleteTarget({ id });
  };
  
  const confirmDeleteBlog = async () => {
    if (!deleteTarget) return;
  
    try {
      const res = await fetch(`/api/blogs/${deleteTarget.id}`, {
        method: "DELETE",
        headers: authHeaders(),
      });
  
      const json = await res.json();
  
      if (json.success) {
        setToast({ message: "Blog deleted successfully.", type: "success" });
        setDeleteTarget(null);
        fetchBlogs(search, selectedDate, page);
      } else {
        setToast({
          message: json.message || "Failed to delete blog.",
          type: "error",
        });
      }
    } catch (e) {
      console.error(e);
      setToast({ message: "Failed to delete blog.", type: "error" });
    }
  };

  return (
    <div>
      {/* Header */}
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-[40px] font-semibold text-[#2DBE6C]">Blogs</h1>
          <p className="mt-2 text-[16px] text-[#687280]">
            Manage and organize your blogs
          </p>
        </div>

        <Link
          href={`/${locale}/admin/blogs/add`}
          className="flex items-center gap-2 rounded-lg bg-[#1F8F50] px-5 py-3 text-white hover:bg-[#2DBE6C]"
        >
          <Plus size={18} />
          Add New
        </Link>
      </div>

      {/* Search + date filter */}
      <div className="mb-6 flex flex-col gap-3 md:flex-row">
        <div className="flex flex-1 items-center gap-2 rounded-xl border border-[#CFEFD9] bg-[#F8FFFA] px-4 py-3">
          <Search size={18} className="text-[#1F8F50]" />
          <input
            placeholder="Search blogs..."
            value={search}
            onChange={(e) => handleSearchChange(e.target.value)}
            className="w-full bg-transparent text-sm outline-none"
          />
        </div>

        <div className="flex items-center gap-2 rounded-xl border border-[#CFEFD9] bg-[#F8FFFA] px-4 py-3">
          <Filter size={18} className="text-[#1F8F50]" />
          <input
            type="date"
            value={selectedDate}
            onChange={(e) => handleDateChange(e.target.value)}
            className="bg-transparent text-sm outline-none"
          />
          {selectedDate && (
            <button
              type="button"
              onClick={() => handleDateChange("")}
              className="text-sm text-[#1F8F50] hover:underline"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Table */}
      {loading ? (
        <div className="flex justify-center py-16">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-green-500 border-t-transparent" />
        </div>
      ) : (
        <>
          <BlogTable data={blogs} locale={locale} onDelete={handleDelete} />

          <Pagination
            page={page}
            totalPages={totalPages}
            onPageChange={setPage}
            total={total}
            pageSize={PAGE_SIZE}
            variant="admin"
          />
        </>
      )}
            <ConfirmModal
        open={!!deleteTarget}
        title="Delete Blog"
        message="Are you sure you want to delete this blog? This action cannot be undone."
        confirmText="Delete"
        cancelText="Cancel"
        isDanger
        onConfirm={confirmDeleteBlog}
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
