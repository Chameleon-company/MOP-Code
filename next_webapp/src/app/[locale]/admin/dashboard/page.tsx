import { LayoutGrid, Folder, ImageIcon, FileText } from "lucide-react";
import AdminStatCard from "@/components/admin/AdminStatsCard";
import AdminRecentActivity from "@/components/admin/AdminRecentActivity";

export default function DashboardPage() {
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
          value="5"
          icon={<LayoutGrid size={32} />}
        />
        <AdminStatCard
          title="Use Cases"
          value="10"
          icon={<Folder size={32} />}
        />
        <AdminStatCard
          title="Gallery Photos"
          value="24"
          icon={<ImageIcon size={32} />}
        />
        <AdminStatCard
          title="Blogs"
          value="15"
          icon={<FileText size={32} />}
        />
      </div>

      {/* Recent Activity */}
      <div className="mt-12">
        <AdminRecentActivity />
      </div>
    </div>
  );
}