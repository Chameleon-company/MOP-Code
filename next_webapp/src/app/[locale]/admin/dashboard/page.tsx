"use client";

import { useEffect, useState } from "react";
import { LayoutGrid, Folder, BookOpen, Images } from "lucide-react";
import AdminStatCard from "@/components/admin/AdminStatsCard";
import AdminRecentActivity from "@/components/admin/AdminRecentActivity";

function getAuthHeaders(): HeadersInit {
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
        const [totalRes, categoryRes, blogsRes, galleryRes] = await Promise.all([
          fetch("/api/statistics/total-count"),
          fetch("/api/statistics/by-category"),
          fetch("/api/blogs?page=1&pageSize=1", { headers }),
          fetch("/api/gallery?page=1&pageSize=1", { headers }),
        ]);

        const totalData = await totalRes.json();
        const categoryData = await categoryRes.json();
        const blogsData = await blogsRes.json();
        const galleryData = await galleryRes.json();

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
        // Keep "—" on error
      } finally {
        setLoading(false);
      }
    }

    fetchStats();
  }, []);

  const displayValue = (value: string) => (loading ? "…" : value);

  return (
    <div className="p-6">
      {/* Title */}
      <h1 className="mb-10 text-[40px] font-semibold leading-[48px] text-[#2DBE6C]">
        Dashboard
      </h1>

      {/* Cards Section */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
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
