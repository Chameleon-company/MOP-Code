"use client";

import Link from "next/link";
import { useParams, usePathname } from "next/navigation";
import {
  Menu,
  X,
  LayoutDashboard,
  FolderOpen,
  Briefcase,
  Image as ImageIcon,
  FileText,
  History,
    Users,
} from "lucide-react";

const menuItems = [
  { label: "Dashboard", path: "/admin/dashboard", icon: LayoutDashboard },
  { label: "Categories", path: "/admin/categories", icon: FolderOpen },
  { label: "Use Cases", path: "/admin/use-cases", icon: Briefcase },
  { label: "Gallery", path: "/admin/gallery", icon: ImageIcon },
  { label: "Contributors", path: "/admin/contributors", icon: Users },
  // { label: "Blogs", path: "/admin/blogs", icon: FileText },
  { label: "Activity History", path: "/admin/activity-history", icon: History },
  { label: "Blogs", path: "/admin/blogs", icon: FileText },
];

type AdminSidebarProps = {
  sidebarOpen: boolean;
  setSidebarOpen: React.Dispatch<React.SetStateAction<boolean>>;
  mobileOpen: boolean;
  setMobileOpen: React.Dispatch<React.SetStateAction<boolean>>;
};

export default function AdminSidebar({
  sidebarOpen,
  setSidebarOpen,
  mobileOpen,
  setMobileOpen,
}: AdminSidebarProps) {
  const pathname = usePathname();
  const params = useParams();
  const locale = params?.locale as string;

  return (
    <>
      {/* Backdrop — mobile/tablet only, shown while the drawer is open */}
      {mobileOpen && (
        <button
          type="button"
          aria-label="Close menu"
          onClick={() => setMobileOpen(false)}
          className="fixed inset-0 z-20 bg-black/50 backdrop-blur-[2px] dark:bg-black/60 lg:hidden"
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-30 w-[190px] bg-[#1F8F50] shadow-sm transition-transform duration-300 ${
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        } lg:static lg:z-auto lg:translate-x-0 lg:transition-all lg:duration-300 ${
          sidebarOpen ? "lg:w-[190px] lg:bg-[#1F8F50]" : "lg:w-[70px] lg:bg-[#F1EFEF]"
        }`}
      >
        <div className="flex items-center px-2 py-3 lg:px-3">
          {/* Close button — mobile drawer only */}
          <button
            type="button"
            onClick={() => setMobileOpen(false)}
            aria-label="Close menu"
            className="flex h-10 w-10 items-center justify-center rounded-lg text-white transition hover:bg-white/20 lg:hidden"
          >
            <X size={20} />
          </button>

          {/* Collapse/expand button — desktop only */}
          <button
            type="button"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            aria-label={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
            className="hidden h-10 w-10 items-center justify-center rounded-lg text-black transition hover:bg-white/20 lg:flex"
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
                onClick={() => setMobileOpen(false)}
                className={`flex items-center gap-2 rounded-lg px-3 py-2 transition-all duration-200 ${
                  sidebarOpen
                    ? "lg:gap-3 lg:px-3 lg:py-2"
                    : "lg:justify-center lg:gap-0 lg:px-0 lg:py-2"
                } ${
                  isActive
                    ? "bg-white text-[#1F8F50]"
                    : sidebarOpen
                    ? "text-white hover:bg-white/20"
                    : "text-white hover:bg-white/20 lg:text-black lg:hover:bg-black/5"
                }`}
              >
                <Icon size={16} className="md:h-[18px] md:w-[18px]" />
                <span
                  className={`text-[12px] font-medium leading-tight md:text-[14px] ${
                    !sidebarOpen ? "lg:hidden" : ""
                  }`}
                >
                  {item.label}
                </span>
              </Link>
            );
          })}
        </nav>
      </aside>
    </>
  );
}
