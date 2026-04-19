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

  const asideBase = `transition-all duration-300 shadow-sm`;
  const asideWidth = sidebarOpen
    ? "w-[120px] md:w-[180px] lg:w-[190px]"
    : "w-[56px] md:w-[70px]";
  const asideBg = sidebarOpen ? "bg-[#1F8F50]" : "bg-[#F1EFEF]";

  return (
    <>
      {/* Mobile + tablet */}
      <aside
        className={`absolute left-0 top-0 bottom-0 z-20 lg:hidden ${asideBase} ${asideWidth} ${asideBg}`}
      >
        <div className="px-2 py-3">
          <button
            type="button"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="flex h-10 w-10 items-center justify-center rounded-lg text-black transition hover:bg-white/20"
          >
            <Menu size={20} />
          </button>
        </div>

        <nav className="space-y-2 px-2 pt-4">
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
                  sidebarOpen ? "gap-2 px-3 py-2" : "justify-center px-0 py-2"
                } ${
                  isActive
                    ? "bg-white text-[#1F8F50]"
                    : sidebarOpen
                    ? "text-black hover:bg-white/20"
                    : "text-black hover:bg-black/5"
                }`}
              >
                <Icon size={16} className="md:h-[18px] md:w-[18px]" />
                {sidebarOpen && (
                  <span className="text-[12px] md:text-[14px] font-medium leading-tight">
                    {item.label}
                  </span>
                )}
              </Link>
            );
          })}
        </nav>
      </aside>

      {/* Desktop only */}
      <aside
        className={`hidden lg:block ${asideBase} ${
          sidebarOpen ? "w-[190px] bg-[#1F8F50]" : "w-[70px] bg-[#F1EFEF]"
        }`}
      >
        <div className="px-3 py-3">
          <button
            type="button"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="flex h-10 w-10 items-center justify-center rounded-lg text-black transition hover:bg-white/20"
          >
            <Menu size={20} />
          </button>
        </div>

        <nav className="space-y-2 px-2 pt-4">
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
                  sidebarOpen ? "gap-3 px-3 py-2" : "justify-center px-0 py-2"
                } ${
                  isActive
                    ? "bg-white text-[#1F8F50]"
                    : sidebarOpen
                    ? "text-black hover:bg-white/20"
                    : "text-black hover:bg-black/5"
                }`}
              >
                <Icon size={16} className="h-[18px] w-[18px]" />
                {sidebarOpen && (
                  <span className="text-[14px] font-medium">{item.label}</span>
                )}
              </Link>
            );
          })}
        </nav>
      </aside>
    </>
  );
}