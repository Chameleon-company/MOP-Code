"use client";

import { useEffect, useState } from "react";
import { LayoutGrid, Folder, BookOpen, Images } from "lucide-react";
import AdminStatCard from "@/components/admin/AdminStatsCard";
import AdminRecentActivity from "@/components/admin/AdminRecentActivity";
import { storage } from "@/utils/storage";
import { apiFetch } from "@/lib/apiFetch";

function getAuthHeaders(): HeadersInit {
  let user: Record<string, any> = {};
  try {
    user = JSON.parse(storage.getItem("user") || "{}");
  } catch {
    user = {};
  }
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

export default function DashboardPage() {
  const [totalUseCases, setTotalUseCases] = useState<string>("—");
  const [totalCategories, setTotalCategories] = useState<string>("—");
  const [totalBlogs, setTotalBlogs] = useState<string>("—");
  const [totalGallery, setTotalGallery] = useState<string>("—");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchStats() {
      try {
        const headers = getAuthHeaders();
        const [totalData, categoryData, blogsData, galleryData] = await Promise.all([
          apiFetch<any>("/api/statistics/total-count"),
          apiFetch<any>("/api/statistics/by-category"),
          apiFetch<any>("/api/blogs?page=1&pageSize=1", { headers }),
          apiFetch<any>("/api/gallery?page=1&pageSize=1", { headers }),
        ]);

        if (totalData.success) {
          setTotalUseCases(String(totalData.total));
        }
        if (categoryData.success) {
          const count = categoryData.data.filter(
            (d: { category: string }) => d.category !== "Uncategorized"
          ).length;
          setTotalCategories(String(count));
        }
        if (blogsData.success) {
          setTotalBlogs(String(blogsData.pagination?.total ?? 0));
        }
        if (galleryData.success) {
          setTotalGallery(String(galleryData.pagination?.total ?? 0));
        }
      } catch {
        // Keep "—" on error; apiFetch already showed a toast
      } finally {
        setLoading(false);
      }
    }

    fetchStats();
  }, []);

  const displayValue = (value: string) => (loading ? "…" : value);

  return (
    <div>
      {/* Title */}
      <h1 className="mb-6 text-2xl font-semibold leading-tight text-emerald-500 sm:text-3xl md:mb-8 md:text-[40px]">
        Dashboard
      </h1>

      {/* Cards Section */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
        <AdminStatCard
          title="Total Categories"
          value={displayValue(totalCategories)}
          icon={<LayoutGrid size={32} />}
        />
        <AdminStatCard
          title="Use Cases"
          value={displayValue(totalUseCases)}
          icon={<Folder size={32} />}
        />
        <AdminStatCard
          title="Total Blogs"
          value={displayValue(totalBlogs)}
          icon={<BookOpen size={32} />}
        />
        <AdminStatCard
          title="Gallery Photos"
          value={displayValue(totalGallery)}
          icon={<Images size={32} />}
        />
      </div>

      {/* Recent Activity */}
      <div className="mt-12">
        <AdminRecentActivity />
      </div>
    </div>
  );
}
