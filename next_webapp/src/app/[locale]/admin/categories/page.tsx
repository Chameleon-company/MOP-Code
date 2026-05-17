"use client";
import ConfirmModal from "@/components/admin/ConfirmModal";
import AdminToast from "@/components/admin/AdminToast";
import ImageHoverPreview from "@/components/admin/ImageHoverPreview";
import TextHoverPreview from "@/components/admin/TextHoverPreview";
import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { Plus, Pencil, Trash2 } from "lucide-react";
import Pagination from "@/components/Pagination";

function getAuthHeaders(): HeadersInit {
  const user = JSON.parse(localStorage.getItem("user") || "{}");
  const userId = user.userId ?? user.id ?? localStorage.getItem("userId") ?? "";
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

export default function CategoriesPage() {
  const { locale } = useParams() as { locale: string };
  const router = useRouter();

  const [categories, setCategories] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  const [deleteTarget, setDeleteTarget] = useState<{ id: number; name: string } | null>(null);
  const [toast, setToast] = useState<{ message: string; type: "success" | "error" } | null>(null);

  useEffect(() => {
    fetchCategories(page);
  }, [page]);

  async function fetchCategories(currentPage: number) {
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams({
        page: String(currentPage),
        pageSize: String(PAGE_SIZE),
      });
      const res = await fetch(`/api/categories?${params}`, {
        headers: getAuthHeaders(),
      });
      const json = await res.json();
      if (res.status === 401) {
        localStorage.removeItem("user");
        router.replace(`/${locale}/login`);
        return;
      }
      if (json.success) {
        setCategories(json.data || []);
        setTotalPages(json.pagination?.totalPages ?? 1);
        setTotal(json.pagination?.total ?? 0);
      } else {
        setError(json.message || "Failed to load categories.");
      }
    } catch {
      setError("Failed to load categories.");
    } finally {
      setLoading(false);
    }
  }

  function handleDelete(id: number, name: string) {
    setDeleteTarget({ id, name });
  }
  
  async function confirmDeleteCategory() {
    if (!deleteTarget) return;
  
    try {
      const res = await fetch(`/api/categories/${deleteTarget.id}`, {
        method: "DELETE",
        headers: getAuthHeaders(),
      });
  
      const json = await res.json();
  
      if (!json.success) {
        setToast({
          message: json.message || "Failed to delete category.",
          type: "error",
        });
        return;
      }
  
      setToast({
        message: "Category deleted successfully.",
        type: "success",
      });
  
      setDeleteTarget(null);
  
      if (categories.length === 1 && page > 1) {
        setPage((p) => p - 1);
      } else {
        fetchCategories(page);
      }
    } catch {
      setToast({
        message: "Failed to delete category.",
        type: "error",
      });
    }
  }

  return (
    <div>
      {/* Header */}
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
          href={`/${locale}/admin/categories/add`}
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

      {/* Table */}
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
              {loading && (
                <tr>
                  <td colSpan={4} className="px-3 py-10 text-center text-[14px] text-[#687280]">
                    Loading...
                  </td>
                </tr>
              )}

              {!loading && categories.map((category) => (
                <tr key={category.id} className="border-b border-black/10">
                  <td className="px-3 py-4">
                  <ImageHoverPreview
                     src={category.cover_img || "/images/category-placeholder.png"}
                     alt={category.category_name}
                  />
                  </td>

                  <td className="px-3 py-4 text-[14px] font-medium text-black">
                    {category.category_name}
                  </td>

                  <td className="px-3 py-4 text-[14px] text-[#687280]">
  <TextHoverPreview text={category.description || "—"} />
</td>

                  <td className="px-3 py-4">
                    <div className="flex items-center gap-2">
                      <Link href={`/${locale}/admin/categories/edit/${category.id}`}>
                        <button className="rounded-lg bg-white p-2 text-[#1F8F50] transition hover:bg-[#DFF7E8]">
                          <Pencil size={16} />
                        </button>
                      </Link>
                      <button
                        onClick={() => handleDelete(category.id, category.category_name)}
                        className="rounded-lg bg-white p-2 text-red-500 transition hover:bg-red-50"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}

              {!loading && categories.length === 0 && (
                <tr>
                  <td colSpan={4} className="px-3 py-10 text-center text-[14px] text-[#687280]">
                    No data available at the moment.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

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
        title="Delete Category"
        message={`Are you sure you want to delete "${deleteTarget?.name}"? This action cannot be undone.`}
        confirmText="Delete"
        cancelText="Cancel"
        isDanger
        onConfirm={confirmDeleteCategory}
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
