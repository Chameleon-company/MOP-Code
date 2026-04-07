"use client";

import Link from "next/link";
import { useParams, usePathname } from "next/navigation";
import {
  Menu,
  LayoutDashboard,
  FolderOpen,
  Briefcase,
  Image as ImageIcon,
  FileText,
} from "lucide-react";

const menuItems = [
  { label: "Dashboard", path: "/admin/dashboard", icon: LayoutDashboard },
  { label: "Categories", path: "/admin/categories", icon: FolderOpen },
  { label: "Use Cases", path: "/admin/use-cases", icon: Briefcase },
  { label: "Gallery", path: "/admin/gallery", icon: ImageIcon },
  { label: "Blogs", path: "/admin/blogs", icon: FileText },
];

type AdminSidebarProps = {
  sidebarOpen: boolean;
  setSidebarOpen: React.Dispatch<React.SetStateAction<boolean>>;
};

export default function AdminSidebar({
  sidebarOpen,
  setSidebarOpen,
}: AdminSidebarProps) {
  const pathname = usePathname();
  const params = useParams();
  const locale = params?.locale as string;

  return (
    <aside
      className={`min-h-screen bg-[#1F8F50] transition-all duration-300 shadow-sm ${
        sidebarOpen ? "w-[190px]" : "w-[70px]"
      }`}
    >
      {/* Menu button */}
      <div className="px-3 py-3">
        <button
          type="button"
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="flex h-10 w-10 items-center justify-center rounded-lg text-black hover:bg-white/20 transition"
        >
          <Menu size={20} />
        </button>
      </div>

      {/* Menu */}
      <nav className="px-2 pt-4 space-y-2">
        {menuItems.map((item) => {
          const href = `/${locale}${item.path}`;
          const isActive = pathname === href;
          const Icon = item.icon;

          return (
            <Link
              key={item.label}
              href={href}
              title={!sidebarOpen ? item.label : ""}
              className={`flex items-center rounded-lg transition-all duration-200 ${
                sidebarOpen
                  ? "gap-3 px-3 py-2"
                  : "justify-center px-0 py-2"
              } ${
                isActive
                  ? "bg-white text-[#1F8F50]"
                  : "text-black hover:bg-white/20"
              }`}
            >
              <Icon size={18} />

              {sidebarOpen && (
                <span className="text-[14px] font-medium">
                  {item.label}
                </span>
              )}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}