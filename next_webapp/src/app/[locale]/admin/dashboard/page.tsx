import { LayoutGrid, Folder, ImageIcon, FileText } from "lucide-react";
import AdminStatCard from "@/components/admin/AdminStatsCard";
import AdminRecentActivity from "@/components/admin/AdminRecentActivity";

export default function DashboardPage() {
  return (
    <div>
      <h1 className="mb-10 text-[40px] font-semibold leading-[48px] text-[#2DBE6C]">
        Dashboard
      </h1>

      <div className="flex flex-wrap gap-4">
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

      <AdminRecentActivity />
    </div>
  );
}